# Experiments

These are the commands used to run the experiments in the paper.
The result files of the experiments are placed at `results/${exp_name}/${model}/version_${version}`.

## multi-task model fusion

```bash
model=ViT-B-32 
# with different learning rate configuration
# TTA training
python scripts/clip_dictmoe.py lr=1e-3 version=0 tta=true evaluate=false num_devices=2
python scripts/clip_dictmoe.py lr=1e-4 version=1 tta=true evaluate=false num_devices=2
python scripts/clip_dictmoe.py lr=5e-5 version=2 tta=true evaluate=false num_devices=2

# Evaluate
python scripts/clip_dictmoe.py lr=1e-3 version=0 tta=false evaluate=true
python scripts/clip_dictmoe.py lr=1e-4 version=1 tta=false evaluate=true
python scripts/clip_dictmoe.py lr=5e-5 version=2 tta=false evaluate=true

model=ViT-L-14
python scripts/clip_dictmoe.py lr=1e-4 version=1 tta=true evaluate=false num_devices=8 model=ViT-L-14

# evaluate
python scripts/clip_dictmoe.py lr=1e-4 version=1 tta=false evaluate=true num_devices=1 model=ViT-L-14
```

## generalization experiments

```bash
model=ViT-B-32
python scripts/clip_dictmoe.py lr=1e-4 version=5 tta=true evaluate=false model=$model \
    num_devices=8 seen_datasets="[SUN397,Cars,RESISC45,DTD,SVHN,GTSRB]"
python scripts/clip_dictmoe.py lr=1e-4 version=6 tta=true evaluate=false model=$model \
    num_devices=8 seen_datasets="[SUN397,Cars,GTSRB,EuroSAT,DTD,MNIST]"

CUDA_VISIBLE_DEVICES=0 python scripts/clip_dictmoe.py lr=1e-4 version=5 tta=false evaluate=true model=$model \
    num_devices=1 seen_datasets="[SUN397,Cars,RESISC45,DTD,SVHN,GTSRB]" &
CUDA_VISIBLE_DEVICES=1 python scripts/clip_dictmoe.py lr=1e-4 version=6 tta=false evaluate=true model=$model \
    num_devices=1 seen_datasets="[SUN397,Cars,GTSRB,EuroSAT,DTD,MNIST]" &
```

## Robust experiments on OOD test data

```bash
# model=ViT-B-32
router_hidden_layers=2
function ood_exp() {
    version=$1
    corruption=$2
    python scripts/clip_dictmoe.py lr=1e-4 version=$version \
        tta=true evaluate=false model=$model exp_name=$exp_name\
        num_devices=2 test_datasets="[Cars,EuroSAT,RESISC45,GTSRB]" corruption=$corruption router_hidden_layers=$router_hidden_layers

    python scripts/clip_dictmoe.py version=$version \
        tta=false evaluate=true model=$model exp_name=$exp_name\
        num_devices=1 test_datasets="[Cars,EuroSAT,RESISC45,GTSRB]" corruption=$corruption router_hidden_layers=$router_hidden_layers
}

model=ViT-B-32 # ViT-B-16
exp_name=clip_dictmoe
CUDA_VISIBLE_DEVICES=0,1 ood_exp 11 null &> 11.log &
CUDA_VISIBLE_DEVICES=2,3 ood_exp 12 motion_blur &> 12.log &
CUDA_VISIBLE_DEVICES=4,5 ood_exp 13 impulse_noise &> 13.log &
CUDA_VISIBLE_DEVICES=6,7 ood_exp 14 gaussian_noise &> 14.log &
CUDA_VISIBLE_DEVICES=1,0 ood_exp 15 pixelate &> 15.log &
CUDA_VISIBLE_DEVICES=3,2 ood_exp 16 spatter &> 16.log &
CUDA_VISIBLE_DEVICES=5,4 ood_exp 17 contrast &> 17.log &
CUDA_VISIBLE_DEVICES=7,6 ood_exp 18 jpeg_compression &> 18.log &


model=ViT-B-32 # ViT-B-16
router_hidden_layers=0
for init_lambda in $(seq 0.1 0.1 1)
do
    exp_name=clip_dictmoe_init_lambda-${init_lambda}_router-${router_hidden_layers}
    CUDA_VISIBLE_DEVICES=0,1 ood_exp 11 null &> 11.log &
    CUDA_VISIBLE_DEVICES=2,3 ood_exp 12 motion_blur &> 12.log &
    CUDA_VISIBLE_DEVICES=4,5 ood_exp 13 impulse_noise &> 13.log &
    CUDA_VISIBLE_DEVICES=6,7 ood_exp 14 gaussian_noise &> 14.log &
    CUDA_VISIBLE_DEVICES=1,0 ood_exp 15 pixelate &> 15.log &
    CUDA_VISIBLE_DEVICES=3,2 ood_exp 16 spatter &> 16.log &
    CUDA_VISIBLE_DEVICES=5,4 ood_exp 17 contrast &> 17.log &
    CUDA_VISIBLE_DEVICES=7,6 ood_exp 18 jpeg_compression &> 18.log &
    wait
done
```

### Ties-Merging

```bash
model=ViT-B-32
# model=ViT-B-16
function ties_merging_exp() {
    corruption=$1
    python scripts/clip_ties_merging.py model=$model corruption=$corruption test_datasets="[Cars,EuroSAT,RESISC45,GTSRB]" batch_size=64 num_workers=8
}

CUDA_VISIBLE_DEVICES=0 ties_merging_exp null &> tm-1.log &
CUDA_VISIBLE_DEVICES=1 ties_merging_exp motion_blur &> tm-2.log &
CUDA_VISIBLE_DEVICES=2 ties_merging_exp impulse_noise &> tm-3.log &
CUDA_VISIBLE_DEVICES=3 ties_merging_exp gaussian_noise &> tm-4.log &
CUDA_VISIBLE_DEVICES=4 ties_merging_exp pixelate &> tm-5.log &
CUDA_VISIBLE_DEVICES=5 ties_merging_exp spatter &> tm-6.log &
CUDA_VISIBLE_DEVICES=6 ties_merging_exp contrast &> tm-7.log &
CUDA_VISIBLE_DEVICES=7 ties_merging_exp jpeg_compression &> tm-8.log &
```

### Task arithematic

```bash
# model=ViT-B-32
model=ViT-B-16
function task_arithmetic_exp() {
    corruption=$1
    python scripts/clip_task_arithmetic.py model=$model corruption=$corruption test_datasets="[Cars,EuroSAT,RESISC45,GTSRB]" batch_size=64 num_workers=8
}

CUDA_VISIBLE_DEVICES=0 task_arithmetic_exp null &> ta-1.log &
CUDA_VISIBLE_DEVICES=1 task_arithmetic_exp motion_blur &> ta-2.log &
CUDA_VISIBLE_DEVICES=2 task_arithmetic_exp impulse_noise &> ta-3.log &
CUDA_VISIBLE_DEVICES=3 task_arithmetic_exp gaussian_noise &> ta-4.log &
CUDA_VISIBLE_DEVICES=4 task_arithmetic_exp pixelate &> ta-5.log &
CUDA_VISIBLE_DEVICES=5 task_arithmetic_exp spatter &> ta-6.log &
CUDA_VISIBLE_DEVICES=6 task_arithmetic_exp contrast &> ta-7.log &
CUDA_VISIBLE_DEVICES=7 task_arithmetic_exp jpeg_compression &> ta-8.log &
```


