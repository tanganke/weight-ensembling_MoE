from _common import *

log = logging.getLogger(__name__)

from clip_checkpoint_path import (
    CHECKPOINT_DIR,
    finetuned_model_path,
    pretrained_model_path,
    sam_retraining_model_path,
)

from src.clip_eval import eval_single_dataset
from src.ties_merging_utils import *


@hydra.main(config_path=str(CONFIG_DIR), config_name="clip_default", version_base=None)
def main(cfg: DictConfig):
    cfg.save = str(CHECKPOINT_DIR / cfg.model)
    cfg.data_location = str(DATA_DIR)
    model = cfg.model

    pretrained_checkpoint = pretrained_model_path(model)
    ft_checks: List[StateDict] = [
        torch.load(finetuned_model_path(model, dataset_name), map_location="cpu").state_dict()
        for dataset_name in tqdm(cfg.test_datasets, "load finetuned checkpoints")
    ]
    ptm_check: StateDict = torch.load(pretrained_checkpoint, map_location="cpu").state_dict()
    check_parameterNamesMatch(ft_checks + [ptm_check])

    remove_keys = []
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    tv_flat_checks = flat_ft - flat_ptm
    assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
    assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]) for i in range(len(ft_checks))])

    results = {"scaling_coef": [], "dataset": [], "acc": []}

    K = 20
    merge_func = "dis-sum"

    for scaling_coef_ in np.linspace(0, 1, 11):
        merged_tv = ties_merging(
            tv_flat_checks,
            reset_thresh=K,
            merge_func=merge_func,
        )
        merged_check = flat_ptm + scaling_coef_ * merged_tv
        merged_state_dict = vector_to_state_dict(merged_check, ptm_check, remove_keys=remove_keys)

        image_encoder: nn.Module = torch.load(pretrained_checkpoint)
        image_encoder.load_state_dict(merged_state_dict, strict=False)

        Total_ACC = 0.0
        for dataset in cfg.test_datasets:
            metrics = eval_single_dataset(image_encoder, dataset, cfg)
            Total_ACC += metrics["top1"]
            log.info(str(dataset) + ":" + str(metrics))

            results["scaling_coef"].append(scaling_coef_)
            results["dataset"].append(dataset)
            results["acc"].append(metrics["top1"])

        log.info("Final: " + "Avg ACC:" + str(Total_ACC / len(cfg.test_datasets)))

    save_dir = RESULTS_DIR / cfg.model
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(save_dir / f"ties_merging-corruption={cfg.corruption}.csv", index=False)


if __name__ == "__main__":
    main()
