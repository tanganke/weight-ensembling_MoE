from _common import *

log = logging.getLogger(__name__)

from src.clip_eval import eval_single_dataset
from src.task_vectors import StateDict, TaskVector, state_dict_mean
from src.ties_merging_utils import check_parameterNamesMatch

from clip_checkpoint_path import CHECKPOINT_DIR, finetuned_model_path, pretrained_model_path


@hydra.main(config_path=str(CONFIG_DIR), config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg.save = str(CHECKPOINT_DIR / cfg.model)
    cfg.data_location = str(DATA_DIR)
    model = cfg.model

    log.info("load finetuned models")
    ft_checks: List[StateDict] = [torch.load(finetuned_model_path(model, dataset_name)).state_dict() for dataset_name in tqdm(cfg.datasets)]
    check_parameterNamesMatch(ft_checks)

    mean_state_dict = state_dict_mean(ft_checks)
    image_encoder: nn.Module = torch.load(pretrained_model_path(model))
    image_encoder.load_state_dict(mean_state_dict)

    results = {"dataset": [], "acc": []}

    accs = []
    for dataset in cfg.datasets:
        metrics = eval_single_dataset(image_encoder, dataset, cfg)
        log.info(str(dataset) + ":" + str(metrics.get("top1") * 100) + "%")
        acc = metrics.get("top1")
        accs.append(metrics.get("top1") * 100)

        results["dataset"].append(dataset)
        results["acc"].append(acc)

    log.info("Avg ACC:" + str(np.mean(accs)) + "%")

    os.makedirs(RESULTS_DIR / cfg.model, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / cfg.model / "averaging.csv", index=False)


if __name__ == "__main__":
    main()
