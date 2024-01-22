<div align="center">

# TinyEDM 🔥

<a href="https://pytorch.org/get-started/locally/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.10+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch -ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-89b8cd?style=for-the-badge&labelColor=gray"></a>
![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)

[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

This a an unofficial PyTorch (Lightning) implementation of EDM [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) and [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696).
</div>

- [x] Config G.
- [ ] Post-hoc EMA.
## Installation

```
git clone https://github.com/YichengDWu/tinyedm.git
cd tinyedm && pip install .
```

## Train

```python
python experiments/train.py --config-name=mnist
python experiments/train.py --config-name=cifar10
```

## Generate
```python
python /src/tinyedm/generate.py \
    --ckpt_path /path/to/checkpoint.ckpt \
    --use_ema \
    --output_dir /path/to/output \
    --num_samples 5000 \
    --image_size 32 \
    --num_classes 10 \
    --batch_size 128 \
    --num_workers 16 \
    --num_steps 32
```