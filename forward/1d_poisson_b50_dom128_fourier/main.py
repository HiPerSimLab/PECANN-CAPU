import torch

from config import Config
from core import Trainer
from utils import _torch_device, _ensure_dirs

def main():
    torch.set_default_dtype(torch.float64)
    cfg = Config()
    print("Using device:", _torch_device())
    _ensure_dirs(cfg)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

