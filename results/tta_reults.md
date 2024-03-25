# More details about Test-Time Adaptation (TTA)

Here we are going to show some results about fine-tuning a single model via test-time adaptation training (entropy minimization).
We find it is unstable to full fine-tune a model via test-time adaptation training.

## Single-Task Experiments

In this section, we are going to show some results about single-task models (task-specific CLIP-ViT-B/32 models) TTA on the test dataset.


1. task=Cars, learning rate = 1e-5
  <img width="558" alt="image" src="https://github.com/tanganke/weight-ensembling_MoE/assets/33046358/6dcc95bc-7657-4eec-9042-4c8175b464fb">
  
2. task=Cars, learning rate = 5e-6
  <img width="546" alt="image" src="https://github.com/tanganke/weight-ensembling_MoE/assets/33046358/059ad134-9725-4d0d-b879-8f8443f6e63b">
3. tsak=SVHN, learning rate = 5e-6
  <img width="561" alt="image" src="https://github.com/tanganke/weight-ensembling_MoE/assets/33046358/57e408bb-1b76-4a04-9ee1-eca785c87277">
 
5. task=Cars, learning rate = 1e-6
  <img width="555" alt="image" src="https://github.com/tanganke/weight-ensembling_MoE/assets/33046358/a7e1fb40-8633-4118-a94b-073cf3be2af8">

## Multi-Task Experiments

Here we first obtain a merged model using weight averaging, then full fine-tuning its parameters via test-time adaptation training:

1. overall loss and accuracy:
  <img width="722" alt="image" src="https://github.com/tanganke/weight-ensembling_MoE/assets/33046358/7c28941a-a9a3-42e4-a713-92bd70481b56">

2. entropy loss on each task
  <img width="1102" alt="image" src="https://github.com/tanganke/weight-ensembling_MoE/assets/33046358/d4cf56e8-37cf-47cf-9c3a-4478e1c8d8dc">
  <img width="725" alt="image" src="https://github.com/tanganke/weight-ensembling_MoE/assets/33046358/7c4bdc92-bc75-4101-86a6-409a07d8f854">

4. accuracy on each task
  <img width="1094" alt="image" src="https://github.com/tanganke/weight-ensembling_MoE/assets/33046358/a5f6d447-3dda-4654-946f-347150daaad9">
  <img width="724" alt="image" src="https://github.com/tanganke/weight-ensembling_MoE/assets/33046358/2a0bffe2-1dd0-49d7-a893-4ac72161a41a">


