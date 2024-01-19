import yaml
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch

def load_model(ckpt_path, config_path):
    yaml_config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    cfg = OmegaConf.create(yaml_config)
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(ckpt_path))
    return model

def main(ckpt_path, config_path, output_dir, num_samples, num_classes):
    model = load_model(ckpt_path, config_path)
    
    # load model
    # load checkpoint
    checkpoint = torch.load(ckpt_path)
    
    # sample noise
    # sample class labels
    # generate images
    # save images
    pass