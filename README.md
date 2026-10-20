


# Learning project for flow matching training on images  



## prepare simplified data  

- make 64x64 center crops  
- inherit the npy format of the tutorial we are following for now  

```
uv run python miniature_fm/preprocess_pngs.py  
uv run python miniature_fm/prepare_training_data.py  
```  

## launch  

```
cd miniature_fm  
uv run python train.py  
```  