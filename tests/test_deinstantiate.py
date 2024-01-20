from tinyedm.utils import deinstantiate
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

def test_deinstantiate():
    initialize(config_path="../experiments/conf", job_name="test")
    cfg = compose(config_name="cifar10")
    model: nn.Module = instantiate(cfg.model)

    cfg_dict = deinstantiate(model)
    model_reinstantiated = instantiate(OmegaConf.create(cfg_dict))
    assert isinstance(model_reinstantiated, type(model))
    model_reinstantiated.load_state_dict(model.state_dict(), strict=True)