from pathlib import Path
from typing import Literal

import torch
import yaml 
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Config(BaseModel):

    model_name: str = "Qwen/Qwen3-0.6B-Base"
    freeze_backbone: bool = False

    dataset_name: str = "argilla/ultrafeedback-binarized-preference-cleaned"
    dataset_split: str = "train"
    samples: int = Field(default=100,gt=0)
    rollout_per_prompt: int = Field(32,gt=0, le=100)
    lr: float = 0.01

    max_steps: int = 20
    max_tokens: int = 5500

    # training
    batch_size: int = Field(default=2,gt=0)
    grad_accum_steps: int = Field(default=8, gt=0)
    epochs: int = Field(default=3, gt=0)
    lr: float = Field(default=5e-5, gt=0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    eval_interval: int = Field(default=25, ge=0)
    lr_scheduler: Literal["linear_decay", "warmup_only"] = "linear_decay"
    seed: int = 123


    device: str = "cpu"
    device_id: int = Field(default=0, ge=0)

    use_wandb: bool = True
    skip_demo: bool = False

    @model_validator(mode="after")
    def validate_config(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA not available")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS not available")
        return self

    def get_device(self) -> str:
        """Return the device string for PyTorch."""
        if self.device == "cuda" and torch.cuda.is_available():
            return f"cuda:{self.device_id}"
        return self.device


def load_config(config_path: str | Path) -> Config:
    with open(config_path) as f: 
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)

def save_config(config: Config, path: str | Path):
    with open(path, "w") as f:
        yaml.safe_dump(config.model_dump(), f)