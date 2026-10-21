import modal
import os

# Create Modal app
app = modal.App("miniature-fm-training")

# Create/reference the Modal volume (mirrors local structure)
volume = modal.Volume.from_name("2026-10-19-miniature-fm", create_if_missing=True)

# Define the container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "numpy",
        "pillow",
        "tqdm",
        "matplotlib",
        "pytorch-optimizer>=4.0.0",
    )
    .add_local_python_source("miniature_fm")
)

def save_images(cfg, dataset, generated, epoch, num_samples):
    """Save comparison images of training data vs generated samples"""
    import numpy as np
    import matplotlib.pyplot as plt

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    train_imgs = dataset[indices][0].permute(0, 2, 3, 1).numpy()

    gen_imgs = [
        (g.transpose(0, 2, 3, 1).astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        for g in generated]

    fig, axes = plt.subplots(1 + len(gen_imgs), num_samples, figsize=(num_samples * 3, 3*(1+len(gen_imgs))))
    for i in range(num_samples):
        axes[0, i].imshow(train_imgs[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Training Data', fontsize=12, loc='left')
        for j in range(1, len(gen_imgs) + 1):
            axes[j, i].imshow(gen_imgs[j - 1][i])
            axes[j, i].axis('off')
            if i == 0:
                axes[j, i].set_title(f'Generated {j}', fontsize=12, loc='left')

    plt.suptitle(f'Epoch {epoch}', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{cfg.plot_dir}/comparison_{epoch:03d}.png', dpi=120)
    plt.close(fig)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,  # 1 hour timeout
    volumes={"/root/2026-10-19-miniature-fm/volumes": volume},
)
def train_on_modal(experiment_tag: str = "teensy"):
    """Train the model on Modal GPU"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from miniature_fm.model import DiT
    from miniature_fm.config import Config
    from torch.utils.data import DataLoader, TensorDataset
    from pytorch_optimizer import get_wsd_schedule
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for Modal
    import matplotlib.pyplot as plt
    import json
    from datetime import datetime

    # Update config for Modal/GPU execution
    cfg = Config()
    cfg.experiment_tag = experiment_tag

    # Generate experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.experiment_dir = f"/root/2026-10-19-miniature-fm/volumes/models/{cfg.experiment_tag}_{timestamp}"
    cfg.ckpt_dir = os.path.join(cfg.experiment_dir, "checkpoints")
    cfg.plot_dir = os.path.join(cfg.experiment_dir, "plots")
    cfg.loss_plot_dir = os.path.join(cfg.experiment_dir, "loss_plots")

    # Update data path to Modal volume path
    cfg.data_path = "/root/2026-10-19-miniature-fm/volumes/data/intermediates/visual_poetry/step2_numpy_array.npy"

    # Enable GPU optimizations
    torch.set_float32_matmul_precision('high')

    # Create output directories
    if cfg.save_ckpt_interval > 0 and cfg.ckpt_dir:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
    if cfg.save_plot_interval > 0 and cfg.plot_dir:
        os.makedirs(cfg.plot_dir, exist_ok=True)
    if cfg.loss_plot_dir:
        os.makedirs(cfg.loss_plot_dir, exist_ok=True)
        metrics_log_path = f'{cfg.loss_plot_dir}/metrics.jsonl'

    # Use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = DiT(cfg).to(device)
    optimizer = optim.AdamW(
        get_decay_params(model, cfg.weight_decay),
        lr=cfg.lr,
        fused=True  # Enable fused optimizer for GPU
    )

    scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=10,
        num_stable_steps=30,
        num_decay_steps=10,
        min_lr_ratio=0.0,
        cooldown_type="1-sqrt",
    )

    # Load data
    images = torch.from_numpy(np.load(cfg.data_path)).permute(0, 3, 1, 2).contiguous()
    dataset = TensorDataset(images)

    # Split train/val
    num_val = cfg.batch_size * 2
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Modal containers work better with num_workers=0
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f'Training data loaded: {len(train_dataset)} images in {len(train_loader)} batches')
    print(f'Validation data loaded: {len(val_dataset)} images in {len(val_loader)} batches')

    def calc_loss(y: torch.Tensor):
        B, C, H, W = y.shape
        y = y.to(device, non_blocking=True).float() / 255.0
        z = torch.randn(B, C, H, W, device=device)
        t = torch.rand(B, device=device)
        v = y - z
        x = z + v * t.view(B, 1, 1, 1)
        v_pred = model(x, t)
        loss = F.mse_loss(v_pred, v)
        return loss

    # Enable torch.compile for GPU
    calc_loss_c = torch.compile(calc_loss)

    train_losses = []
    val_losses = []
    learning_rates = []

    for epoch in range(cfg.epochs):
        total_loss = 0

        model.train()
        for (y,) in train_loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = calc_loss_c(y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        # Validation
        val_total_loss = 0
        model.eval()
        with torch.no_grad():
            for (y,) in val_loader:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss = calc_loss_c(y)
                val_total_loss += loss.item()
        avg_val_loss = val_total_loss / len(val_loader)

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)

        # Log metrics to JSONL
        if cfg.loss_plot_dir:
            with open(metrics_log_path, 'a') as f:
                json.dump({
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'val_loss': avg_val_loss,
                    'learning_rate': current_lr
                }, f)
                f.write('\n')

        print(f"Epoch {epoch:03d}/{cfg.epochs} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        if cfg.save_ckpt_interval > 0 and cfg.ckpt_dir and (epoch % cfg.save_ckpt_interval == 0 or epoch == cfg.epochs - 1):
            torch.save(model.state_dict(), f"{cfg.ckpt_dir}/epoch_{epoch:03d}.pt")

        if cfg.save_plot_interval > 0 and cfg.plot_dir and (epoch % cfg.save_plot_interval == 0 or epoch == cfg.epochs - 1):
            model.eval()
            num_samples = 8
            steps = 30
            x = torch.randn(num_samples, cfg.in_channels, cfg.img_size, cfg.img_size, device=device)
            procs = []
            dt = 1.0 / steps
            with torch.no_grad():
                for i in range(steps):
                    t = torch.ones(num_samples, device=device) * i * dt
                    v_pred = model(x, t)
                    x = x + v_pred * dt
                    procs.append(x.cpu().clone().numpy())
            print(f"Generated {num_samples} samples")
            save_images(cfg, dataset, procs, epoch, num_samples)

    # Save final loss plot
    if cfg.loss_plot_dir:
        save_loss_plot(cfg, train_losses, val_losses, learning_rates)
        print(f"Saved final loss plot to {cfg.loss_plot_dir}/losses_final.png")

    # Commit volume changes
    volume.commit()

    print("Training complete!")
    print(f"Experiment directory: {cfg.experiment_dir}")

    # Return the run identifier for syncing
    return f"{cfg.experiment_tag}_{timestamp}"


def get_decay_params(model, weight_decay: float):
    """Helper function to separate decay/no-decay parameters"""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim < 2 or name.endswith(".bias") or name == "pos_embed":
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]


def save_loss_plot(cfg, train_losses, val_losses, learning_rates):
    """Save loss and learning rate plots"""
    import matplotlib.pyplot as plt

    epochs_range = range(len(train_losses))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, train_losses, 'b-', linewidth=1.5, label='Train')
    ax1.plot(epochs_range, val_losses, 'r-', linewidth=1.5, label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, learning_rates, 'g-', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{cfg.loss_plot_dir}/losses_final.png', dpi=100)
    plt.close(fig)


@app.local_entrypoint()
def main(experiment_tag: str = "teensy"):
    """Local entrypoint to run training on Modal"""
    print(f"Starting Modal training run with experiment_tag: {experiment_tag}")
    run_id = train_on_modal.remote(experiment_tag)
    print("\n" + "="*70)
    print(f"✓ Training complete! Run ID: {run_id}")
    print("="*70)
    print(f"\nDownload results:")
    print(f"  uv run modal run miniature_fm/sync_from_modal.py --run-id {run_id}")
    print()
