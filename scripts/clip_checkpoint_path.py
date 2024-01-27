from _common import *

log = logging.getLogger(__name__)

CHECKPOINT_DIR = CACHE_DIR / "checkpoints" / "task_vectors_checkpoints"
MODELS = ["ViT-B-16", "ViT-B-32", "ViT-L-14"]
DATASETS = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]


def pretrained_model_path(model_name: str) -> Path:
    """
    This function generates the path for the pretrained model.

    Parameters:
        model_name (str): The name of the pretrained model.

    Returns:
        Path: The path of the pretrained model.
    """
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    path = CHECKPOINT_DIR / model_name / "zeroshot.pt"
    assert path.is_file(), f"Pretrained model not found at {path}"
    return path


def finetuned_model_path(model_name: str, dataset_name: str) -> Path:
    """
    This function generates the path for the fine-tuned model.

    Parameters:
        model_name (str): The name of the model.
        dataset_name (str): The name of the dataset.

    Returns:
        Path: the path of the fine-tuned model.
    """
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    if dataset_name not in DATASETS:
        log.warning(f"Unknown dataset {dataset_name}")
    path = CHECKPOINT_DIR / model_name / dataset_name / "finetuned.pt"
    assert path.is_file(), f"Finetuned model not found at {path}"
    return path


def sam_retraining_model_path(model_name: str, dataset_name: str):
    if model_name not in MODELS:
        log.warning(f"Unknown model {model_name}")
    if dataset_name not in DATASETS:
        log.warning(f"Unknown dataset {dataset_name}")
    path = PROJECT_DIR / "logs" / "sam_retraining" / model_name / dataset_name / "version_1" / "checkpoints" / "image_encoder_latest.pt"
    assert path.is_file(), f"Finetuned model not found at {path}"
    return path


def main():
    for model in MODELS:
        pretrained_model_path(model)
        for dataset in DATASETS:
            finetuned_model_path(model, dataset)


if __name__ == "__main__":
    main()

__all__ = [n for n in globals().keys() if not n.startswith("_") and n not in ["log", "main"]]
