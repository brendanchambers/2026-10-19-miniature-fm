from dataclasses import dataclass
from datetime import datetime
import os

@dataclass
class Config:

    # Model
    hidden_dim: int = 16 # 512
    emb_hidden_dim:int = 8 # 128
    num_heads: int = 8
    num_layers: int = 6
    patch_size: int = 4
    dropout: float = 0.1

    # Training
    data_path: str = "../volumes/data/intermediates/visual_poetry/step2_numpy_array.npy"
    batch_size: int = 64  # Reduced for CPU training
    lr: float = 2e-3 # 2e-4
    weight_decay: float = 0 # 1e-2
    epochs: int = 5 # 200

    # Experiment organization
    experiment_tag: str = "teensy"
    experiment_dir: str = None  # Auto-generated in __post_init__

    save_ckpt_interval: int = 0 # set to 0 to disable

    # You should not change the following parameters
    #   because the cat dataset has a fixed size of 64x64x3
    img_size: int = 64
    in_channels: int = 3

    def __post_init__(self):
        if self.experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = f"../volumes/models/{self.experiment_tag}_{timestamp}"

        self.ckpt_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.plot_dir = os.path.join(self.experiment_dir, "plots")
        self.loss_plot_dir = os.path.join(self.experiment_dir, "loss_plots")

