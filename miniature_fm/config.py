from dataclasses import dataclass

@dataclass
class Config:

    # Model
    hidden_dim: int = 64 # 512
    emb_hidden_dim:int = 32 # 128
    num_heads: int = 8
    num_layers: int = 6
    patch_size: int = 4
    dropout: float = 0.1

    # Training
    data_path: str = "../volumes/data/intermediates/visual_poetry/step2_numpy_array.npy"
    batch_size: int = 64  # Reduced for CPU training
    lr: float = 2e-4
    weight_decay: float = 1e-2
    epochs: int = 50 # 200

    ckpt_dir: str = '../volumes/models/visual_poetry/checkpoints'
    save_ckpt_interval: int = 0 # set to 0 to disable

    plot_dir: str = '../volumes/models/visual_poetry/plots'
    save_plot_interval: int = 25 # set to 0 to disable

    loss_plot_dir: str = '../volumes/models/visual_poetry/loss_plots'
    save_loss_plot_interval: int = 5 # set to 0 to disable

    # You should not change the following parameters
    #   because the cat dataset has a fixed size of 64x64x3
    img_size: int = 64
    in_channels: int = 3

