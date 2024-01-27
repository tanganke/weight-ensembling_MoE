from _common import *

log = logging.getLogger(__name__)

from collections import defaultdict

from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
    sam_retraining_model_path,
)

from src.clip_eval import eval_single_dataset, eval_single_dataset_preprocess_head
from src.heads import get_classification_head
from src.task_vectors import StateDict, TaskVector
from src.task_wise_fusion import check_parameterNamesMatch
from src.utils import timeit_context


class Program:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        cfg.save = str(CHECKPOINT_DIR / cfg.model)
        cfg.data_location = str(DATA_DIR)

        self.result_path = RESULTS_DIR / cfg.model / "individuals.csv"
        self.fabric = L.Fabric(accelerator="cuda", devices=1)
        self.fabric.launch()

    def run(self):
        self.load_models()

        self.eval_individuals()

    def eval_individuals(self):
        results = defaultdict(list)

        self.model = self.pretrained_model
        _result = defaultdict(list)
        self.eval_model_on_datasets(epoch_idx=0, results=_result)
        results["model"].append("pretrained")
        for dataset_name, acc in zip(_result["dataset"], _result["acc"]):
            results[dataset_name].append(acc)
        print(df := pd.DataFrame(results))
        df.to_csv(self.result_path, index=False)

        for dataset_name, image_encoder in track(
            zip(self.cfg.datasets, self.finetuned_models),
            "evaluating finetuned models",
        ):
            self.model = image_encoder
            _result = defaultdict(list)
            self.eval_model_on_datasets(epoch_idx=0, results=_result)
            results["model"].append(dataset_name)
            for dataset_name, acc in zip(_result["dataset"], _result["acc"]):
                results[dataset_name].append(acc)
        print(df := pd.DataFrame(results))
        df.to_csv(self.result_path, index=False)

    def eval_model_on_datasets(
        self,
        epoch_idx: int,
        results: Dict[str, List[float]],
    ):
        self.model.eval()

        Total_ACC = 0
        for dataset_name in self.cfg.datasets:
            classification_head = self.classification_heads[dataset_name]
            metrics = eval_single_dataset_preprocess_head(self.model, classification_head, dataset_name, self.cfg)
            Total_ACC += metrics["top1"]
            log.info("Eval: init: " + " dataset: " + str(dataset_name) + " ACC: " + str(metrics["top1"]))

            if results is not None:
                results["epoch"].append(epoch_idx)
                results["dataset"].append(dataset_name)
                results["acc"].append(metrics["top1"])

        log.info("Eval: init: " + " Avg ACC:" + str(Total_ACC / len(self.cfg.datasets)) + "\n")

    def load_models(self):
        cfg = self.cfg

        if cfg.sam_retraining:
            log.info("SAM retrained model is used")
            _finetuned_model_path = sam_retraining_model_path
        else:
            _finetuned_model_path = finetuned_model_path
        with timeit_context():
            log.info("load models")
            pretrained_model = torch.load(pretrained_model_path(cfg.model), map_location="cpu")
            finetuned_models = [
                torch.load(_finetuned_model_path(cfg.model, dataset_name), map_location="cpu")
                for dataset_name in track(cfg.datasets, "loading finetuned models")
            ]

        self.pretrained_model = pretrained_model
        self.finetuned_models = finetuned_models
        self.classification_heads = {dataset_name: get_classification_head(cfg, dataset_name).cuda() for dataset_name in cfg.datasets}


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    (program := Program(cfg)).run()


if __name__ == "__main__":
    main()
