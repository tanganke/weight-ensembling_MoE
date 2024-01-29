from _common import *

from src.adamerging import softmax_entropy
from src.datasets.common import maybe_dictionarize

log = logging.getLogger(__name__)

from collections import defaultdict
from typing import cast

import lightning as L
import open_clip.model
from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
)
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.wrappers import _FabricModule
from torch.utils.data import DataLoader

from src.clip_eval import eval_single_dataset
from src.heads import get_classification_head
from src.modeling import ClassificationHead, ImageEncoder
from src.module.dict_moe import DictMoE
from src.module.utils import get_by_name, print_trainable_parameters, set_by_name
from src.task_vectors import StateDict, TaskVector, state_dict_mean
from src.ties_merging_utils import check_parameterNamesMatch
from src.utils import timeit_context


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if cfg.model is None:
            raise ValueError("model must be specified")

        self.result_dir = RESULTS_DIR / cfg.exp_name / cfg.model
        if cfg.version is not None:
            self.result_dir /= f"version_{cfg.version}"
        self.result_dir.mkdir(exist_ok=True, parents=True)
        log.info(f'files will save to {self.result_dir}')
        # save `cfg` to result_dir`
        self.result_path = self.result_dir / "results.csv"

        self.fabric = L.Fabric(
            accelerator="cuda",
            devices=cfg.num_devices,
            strategy="ddp"
            # strategy=self._fsdp_strategy() if cfg.model == "ViT-L-14" else "ddp",
        )
        self.fabric.launch()

    def _fsdp_strategy(self):
        cfg = self.cfg

        policy = {open_clip.model.ResidualAttentionBlock}
        strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            auto_wrap_policy=policy,
            # state_dict_type="full",
            # activation_checkpointing_policy=policy if cfg.model == "ViT-L-14" else None,
        )
        return strategy

    def run(self):
        self.load_model()
        self.load_datasets()

        if self.cfg.tta:
            self.tta()
        if self.cfg.evaluate:
            self.evaluate()

    def tta(self):
        OmegaConf.save(self.cfg, self.result_dir / "tta_config.yaml")

        model = deepcopy(self.model)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=self.cfg.lr)
        model, optimizer = self.fabric.setup(model, optimizer)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

        model.train()
        for step_idx in tqdm(range(1000), "tta training"):
            losses = 0
            for dataset_idx, dataset_name in enumerate(self.cfg.seen_datasets):
                batch = next(self.shuffled_test_loader_iters[dataset_idx])
                batch = maybe_dictionarize(batch)
                x = batch["images"].to(self.fabric.device)  # use images only

                features = model(x)
                logits = self.classification_heads[dataset_name](features)

                loss = softmax_entropy(logits).mean(0)
                losses += loss

            optimizer.zero_grad()
            self.fabric.backward(losses)
            optimizer.step()

            lr_scheduler.step()

            # print(f"step={step_idx}, loss={losses.item()}")

            if (step_idx + 1) % 100 == 0:
                (self.result_dir / "checkpoints").mkdir(exist_ok=True)
                self.fabric.save(self.result_dir / "checkpoints" / f"model_step={step_idx + 1}.ckpt", {"model": model})
                # if self.fabric.is_global_zero:
                # torch.save(model.state_dict(), self.result_dir / "checkpoints" / f"model_step={step_idx + 1}.pt")

    @torch.inference_mode()
    def evaluate(self):
        results = defaultdict(list)

        for step_idx in tqdm([1000, 500], "evaluating", leave=False):
            state_dict = torch.load(self.result_dir / "checkpoints" / f"model_step={step_idx}.ckpt", map_location="cpu")
            if len(state_dict) == 1 and "model" in state_dict:
                state_dict = state_dict["model"]
            model = deepcopy(self.model)
            model.load_state_dict(state_dict)
            model = self.fabric.setup_module(model)
            model.eval()
            results["step"].append(step_idx)

            for dataset_idx, dataset_name in enumerate(
                tqdm(
                    self.cfg.test_datasets,
                    "evaluating datasets",
                    leave=False,
                )
            ):
                test_loader = self.test_loaders[dataset_idx]
                TOTAL_CORRECT = 0
                TOTAL_COUNT = 0
                for batch in (
                    pbar := tqdm(
                        test_loader,
                        f"evaluate {dataset_name}",
                        leave=False,
                    )
                ):
                    batch = maybe_dictionarize(batch)
                    x = batch["images"].to(self.fabric.device)
                    y = batch["labels"].to(self.fabric.device)

                    features = model(x)
                    logits = self.classification_heads[dataset_name](features)
                    preds = logits.argmax(-1)

                    correct = (preds == y).sum().item()
                    TOTAL_CORRECT += correct
                    TOTAL_COUNT += len(y)
                    acc = TOTAL_CORRECT / TOTAL_COUNT
                    pbar.set_postfix_str(f"acc={acc:.2f}")
                results[dataset_name].append(acc)
            (df := pd.DataFrame(results)).to_csv(self.result_path, index=False)
            log.info(df)

    def load_clip_models(self):
        """
        Loads the pretrained CLIP model and the fine-tuned models for each dataset specified in the configuration.
        It first loads the pretrained model from the path specified in the configuration.
        It then loads each fine-tuned model from the path specified in the configuration,
        using the name of the dataset to construct the path.
        Finally, it sets up the classification heads for each dataset, using the configuration and the name of the dataset.

        Side Effects:
            Sets the instance variables `pretrained_model`, `finetuned_models`, and `classification_heads`.
        """
        cfg = self.cfg

        # load pretrained and fine-tuned model
        with timeit_context():
            log.info("load models")
            pretrained_model: ImageEncoder = torch.load(pretrained_model_path(cfg.model), map_location="cpu")
            finetuned_models: List[ImageEncoder] = []
            for dataset_name in track(
                cfg.seen_datasets if cfg.model_seen_datasets is None else cfg.model_seen_datasets,
                "loading finetuned models",
            ):
                log.info(f"loading finetuned model for {dataset_name}")
                finetuned_models.append(
                    torch.load(
                        finetuned_model_path(cfg.model, dataset_name),
                        map_location="cpu",
                    )
                )

        self.pretrained_model = pretrained_model
        self.finetuned_models = finetuned_models
        self.classification_heads = {dataset_name: get_classification_head(cfg, dataset_name).eval() for dataset_name in cfg.test_datasets}
        for m in self.classification_heads.values():
            for p in m.parameters():
                p.requires_grad_(False)
        self.classification_heads = {k: m.to(self.fabric.device) for k, m in self.classification_heads.items()}

    @torch.no_grad()
    def load_model(self):
        self.load_clip_models()
        with timeit_context("Building moe model"):
            model = deepcopy(self.pretrained_model)

            # model fusion
            sd = {}
            base_sd = model.state_dict()
            for name in base_sd.keys():
                sd[name] = base_sd[name]
            for m in self.finetuned_models:
                expert_sd = m.state_dict()
                for name in expert_sd.keys():
                    sd[name] = sd[name] + (expert_sd[name] - base_sd[name]) * self.cfg.init_lambda
            model.load_state_dict(sd)

            # fix all parameters
            for p in model.parameters():
                p.requires_grad_(False)

            for layer_idx in range(model.model.visual.transformer.layers):
                model.model.visual.transformer.resblocks[layer_idx].mlp = DictMoE(
                    hidden_size=model.model.visual.transformer.width,
                    base_model=self.pretrained_model.model.visual.transformer.resblocks[layer_idx].mlp,
                    expert_models=[m.model.visual.transformer.resblocks[layer_idx].mlp for m in self.finetuned_models],
                    init_lambda=self.cfg.init_lambda,
                    fix_base_model_and_experts=True,
                    router_hidden_layers=self.cfg.router_hidden_layers,
                )

            self.model = model
            print_trainable_parameters(model, verbose=True)

    def load_datasets(self):
        """
        Loads the datasets specified in the configuration.

        It first imports the necessary modules and sets up a basic transform for the images.
        It then loads each dataset specified in the configuration, applies the basic transform,
        and sets the location, batch size, and number of workers from the configuration.

        The test dataset from each loaded dataset is added to the list of test datasets.
        It then sets up the data loaders for the test datasets, both with
        and without shuffling, and creates an iterator for each shuffled test loader.

        Side Effects:
            Sets the instance variables `test_datasets`, `test_loaders`, `shuffled_test_loaders`, and
            `shuffled_test_loader_iters`.
        """
        cfg = self.cfg
        cfg.batch_size = cfg.batch_size // cfg.num_devices
        cfg.tta_batch_size = cfg.tta_batch_size // cfg.num_devices
        cfg.eval_batch_size = cfg.eval_batch_size // cfg.num_devices
        print(f"batch_size={cfg.batch_size}, tta_batch_size={cfg.tta_batch_size}, eval_batch_size={cfg.eval_batch_size}")

        if self.cfg.corruption is None:
            from src.datasets.registry import get_dataset
        else:
            from src.datasets.corruption.registry import get_dataset

        cfg = self.cfg

        dataset_kwargs = dict(
            location=cfg.data_location,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        if self.cfg.corruption is not None:
            dataset_kwargs["corruption"] = self.cfg.corruption
        datasets = [
            get_dataset(
                dataset_name,
                self.pretrained_model.val_preprocess,
                **dataset_kwargs,
            )
            for dataset_name in cfg.test_datasets
        ]
        self.test_datasets = [d.test_dataset for d in datasets]
        self.test_loaders = [
            DataLoader(
                d,
                shuffle=False,
                batch_size=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )
            for d in self.test_datasets
        ]
        self.shuffled_test_loaders = [
            DataLoader(
                d,
                shuffle=True,
                batch_size=cfg.tta_batch_size,
                num_workers=cfg.num_workers,
                pin_memory=False,
            )
            for d in self.test_datasets
        ]
        self.shuffled_test_loader_iters = [iter(itertools.cycle(d)) for d in self.shuffled_test_loaders]


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name="clip_dictmoe",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    cfg.save = CACHE_DIR / "checkpoints" / "task_vectors_checkpoints" / cfg.model
    cfg.data_location = str(DATA_DIR)
    program = Program(cfg)
    program.run()


if __name__ == "__main__":
    main()
