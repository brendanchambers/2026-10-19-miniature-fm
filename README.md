


# Learning project for flow matching training on images  



## prepare simplified data  

- make 64x64 center crops  
- inherit the npy format of the tutorial we are following for now  

```
uv run python miniature_fm/preprocess_pngs.py  
uv run python miniature_fm/prepare_training_data.py  
  
uv run python train.py  
```  

## launch  

```
cd miniature_fm  
uv run python train.py  
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