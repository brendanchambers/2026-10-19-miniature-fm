


# Learning project for flow matching training on images  



## prepare simplified data  

- make 64x64 center crops  
- inherit the npy format of the tutorial we are following for now  

```
uv run python miniature_fm/preprocess_pngs.py  
uv run python miniature_fm/prepare_training_data.py  
  
# Upload data to Modal volume (first time or when data changes)
uv run modal run miniature_fm/sync_to_modal.py

```  

## launch  

```
# Train on Modal GPU (automatically syncs latest code)
uv run modal run miniature_fm/train_modal.py

# Download results
uv run modal run miniature_fm/sync_from_modal.py --run-id <run_id>
```  



## Future

```
from pytorch_optimizer import get_wsd_schedule

# Example signature from pytorch-optimizer
scheduler = get_wsd_schedule(
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_stable_steps=50000,
    num_decay_steps=10000,
    min_lr_ratio=0.0,
    cooldown_type="1-sqrt",
)

optimizer.step()
scheduler.step()
```