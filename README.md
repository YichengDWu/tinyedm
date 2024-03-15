<div align="center">

# TinyEDM 🔥

## Analyzing and Improving the Training Dynamics of Diffusion Models

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

```bash
python experiments/train.py --config-name=mnist
python experiments/train.py --config-name=cifar10
```

## ImageNet
To download the ImageNet dataset, follow these steps:
1. Visit the ImageNet website: [http://www.image-net.org/](http://www.image-net.org/)
2. Register for an account and request access for the dataset.
3. Once approved, follow the instructions provided by ImageNet to download the dataset.


### ImageNet Latents

After downloading the ImageNet dataset, extract the files to a directory. When running the feature extraction script, use the `--data-dir` option to specify the path to this directory.

For example:
```bash
python src/tinyedm/datamodules/extract_latents.py --data-dir ./datasets/imagenet/train --out-dir ./datasets/imagenet/latents/train
```

## Generate
```bash
python src/tinyedm/generate.py \
    --ckpt_path /path/to/checkpoint.ckpt \
    --load_ema \
    --output_dir /path/to/output \
    --num_samples 50000 \
    --image_size 32 \
    --num_classes 10 \
    --batch_size 128 \
    --num_workers 16 \
    --num_steps 32
```

## Results


|Dataset | Params       | type | epochs | FID
|----|--------------|:-----:|-----------:|-----------:|
|CIFAR-10| 35.6 M |  unconditional |  1700 | 4.0 |

## Observations

1. Using FP16 mixed precision training on the CIFAR-10 dataset sometimes leads to overflow, so we have adopted bf16 mixed precision, which may result in a loss of accuracy for the model.
2. For the scale factors of skip connections, this implementation uses a small network to learn them, inspired by [ScaleLong: Towards More Stable Training of Diffusion Model via Scaling Network Long Skip Connection
](https://arxiv.org/abs/2310.13545).
3. The use of multi-task learning in the paper did not observe any improvement, or it may be more effective in long-term training. However, I do not have the computational power to verify this.

