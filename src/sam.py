R"""
<h1 align="center"><b>(Adaptive) SAM Optimizer</b></h1>
<h3 align="center"><b>Sharpness-Aware Minimization for Efficiently Improving Generalization</b></h3>
<p align="center">
  <i>~ in Pytorch ~</i>
</p> 
 
--------------

<br>

SAM simultaneously minimizes loss value and loss sharpness. In particular, it seeks parameters that lie in **neighborhoods having uniformly low loss**. SAM improves model generalization and yields [SoTA performance for several datasets](https://paperswithcode.com/paper/sharpness-aware-minimization-for-efficiently-1). Additionally, it provides robustness to label noise on par with that provided by SoTA procedures that specifically target learning with noisy labels.

This is an **unofficial** repository for [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) and [ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks](https://arxiv.org/abs/2102.11600). Implementation-wise, SAM class is a light wrapper that computes the regularized "sharpness-aware" gradient, which is used by the underlying optimizer (such as SGD with momentum). This repository also includes a simple [WRN for Cifar10](example); as a proof-of-concept, it beats the performance of SGD with momentum on this dataset.

<p align="center">
  <img src="img/loss_landscape.png" alt="Loss landscape with and without SAM" width="512"/>  
</p>

<p align="center">
  <sub><em>ResNet loss landscape at the end of training with and without SAM. Sharpness-aware updates lead to a significantly wider minimum, which then leads to better generalization properties.</em></sub>
</p>

<br>

## Usage

It should be straightforward to use SAM in your training pipeline. Just keep in mind that the training will run twice as slow, because SAM needs two forward-backward passes to estime the "sharpness-aware" gradient. If you're using gradient clipping, make sure to change only the magnitude of gradients, not their direction.

```python
from sam import SAM
...

model = YourModel()
base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
...

for input, output in data:

  # first forward-backward pass
  loss = loss_function(output, model(input))  # use this loss for any training statistics
  loss.backward()
  optimizer.first_step(zero_grad=True)
  
  # second forward-backward pass
  loss_function(output, model(input)).backward()  # make sure to do a full forward pass
  optimizer.second_step(zero_grad=True)
...
```

<br>

**Alternative usage with a single closure-based `step` function**. This alternative offers similar API to native PyTorch optimizers like LBFGS (kindly suggested by [@rmcavoy](https://github.com/rmcavoy)):

```python
from sam import SAM
...

model = YourModel()
base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
...

for input, output in data:
  def closure():
    loss = loss_function(output, model(input))
    loss.backward()
    return loss

  loss = loss_function(output, model(input))
  loss.backward()
  optimizer.step(closure)
  optimizer.zero_grad()
...
```

### Training tips
- [@hjq133](https://github.com/hjq133): The suggested usage can potentially cause problems if you use batch normalization. The running statistics are computed in both forward passes, but they should be computed only for the first one. A possible solution is to set BN momentum to zero (kindly suggested by [@ahmdtaha](https://github.com/ahmdtaha)) to bypass the running statistics during the second pass. An example usage is on lines [51](https://github.com/davda54/sam/blob/cdcbdc1574022d3a3c3240da136378c38562d51d/example/train.py#L51) and [58](https://github.com/davda54/sam/blob/cdcbdc1574022d3a3c3240da136378c38562d51d/example/train.py#L58) in [example/train.py](https://github.com/davda54/sam/blob/cdcbdc1574022d3a3c3240da136378c38562d51d/example/train.py):
```python
for batch in dataset.train:
  inputs, targets = (b.to(device) for b in batch)

  # first forward-backward step
  enable_running_stats(model)  # <- this is the important line
  predictions = model(inputs)
  loss = smooth_crossentropy(predictions, targets)
  loss.mean().backward()
  optimizer.first_step(zero_grad=True)

  # second forward-backward step
  disable_running_stats(model)  # <- this is the important line
  smooth_crossentropy(model(inputs), targets).mean().backward()
  optimizer.second_step(zero_grad=True)
```

- [@evanatyourservice](https://github.com/evanatyourservice): If you plan to train on multiple GPUs, the paper states that *"To compute the SAM update when parallelizing across multiple accelerators, we divide each data batch evenly among the accelerators, independently compute the SAM gradient on each accelerator, and average the resulting sub-batch SAM gradients to obtain the final SAM update."* This can be achieved by the following code:
```python
for input, output in data:
  # first forward-backward pass
  loss = loss_function(output, model(input))
  with model.no_sync():  # <- this is the important line
    loss.backward()
  optimizer.first_step(zero_grad=True)
  
  # second forward-backward pass
  loss_function(output, model(input)).backward()
  optimizer.second_step(zero_grad=True)
```
- [@evanatyourservice](https://github.com/evanatyourservice): Adaptive SAM reportedly performs better than the original SAM. The ASAM paper suggests to use higher `rho` for the adaptive updates (~10x larger)

- [@mlaves](https://github.com/mlaves): LR scheduling should be either applied to the base optimizer or you should use SAM with a single `step` call (with a closure):
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=200)
```
- [@AlbertoSabater](https://github.com/AlbertoSabater): Integration with Pytorch Lightning — you can write the `training_step` function as:
```python
def training_step(self, batch, batch_idx):
    optimizer = self.optimizers()

    # first forward-backward pass
    loss_1 = self.compute_loss(batch)
    self.manual_backward(loss_1, optimizer)
    optimizer.first_step(zero_grad=True)

    # second forward-backward pass
    loss_2 = self.compute_loss(batch)
    self.manual_backward(loss_2, optimizer)
    optimizer.second_step(zero_grad=True)

    return loss_1
```
<br>


## Documentation

#### `SAM.__init__`

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `params` (iterable) | iterable of parameters to optimize or dicts defining parameter groups |
| `base_optimizer` (torch.optim.Optimizer) | underlying optimizer that does the "sharpness-aware" update |
| `rho` (float, optional)           | size of the neighborhood for computing the max loss *(default: 0.05)* |
| `adaptive` (bool, optional)       | set this argument to True if you want to use an experimental implementation of element-wise Adaptive SAM *(default: False)* |
| `**kwargs` | keyword arguments passed to the `__init__` method of `base_optimizer` |

<br>

#### `SAM.first_step`

Performs the first optimization step that finds the weights with the highest loss in the local `rho`-neighborhood.

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `zero_grad` (bool, optional) | set to True if you want to automatically zero-out all gradients after this step *(default: False)* |

<br>

#### `SAM.second_step`

Performs the second optimization step that updates the original weights with the gradient from the (locally) highest point in the loss landscape.

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `zero_grad` (bool, optional) | set to True if you want to automatically zero-out all gradients after this step *(default: False)* |

<br>

#### `SAM.step`

Performs both optimization steps in a single call. This function is an alternative to explicitly calling `SAM.first_step` and `SAM.second_step`.

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `closure` (callable) | the closure should do an additional full forward and backward pass on the optimized model *(default: None)* |
"""
import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        base_optimizer (torch.optim.Optimizer): the base optimizer to use for updating the parameters
        rho (float, optional): the learning rate scaling factor (default: 0.05)
        adaptive (bool, optional): whether to use adaptive scaling (default: False)
        **kwargs: additional keyword arguments to pass to the base optimizer

    Example:
        >>> optimizer = SAM(model.parameters(), torch.optim.Adam, lr=0.001)
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Perform the first step of the SAM optimizer.

        This step updates the parameters by adding the scaled gradient to the current values.

        Args:
            zero_grad (bool, optional): whether to zero out the gradients before computing the gradient norm (default: False)
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Perform the second step of the SAM optimizer.

        This step updates the parameters by setting them back to their original values.

        Args:
            zero_grad (bool, optional): whether to zero out the gradients after updating the parameters (default: False)
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (callable, optional): a closure that re-evaluates the model and returns the loss (default: None)
        """
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """
        Compute the gradient norm.

        Returns:
            float: the L2 norm of the gradients
        """
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        """
        Load the optimizer state.

        Args:
            state_dict (dict): the optimizer state
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
