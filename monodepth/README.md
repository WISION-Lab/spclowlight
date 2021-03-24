# Monocular Depth Estimation on noisy SPAD images

This implements training of monocular depth estimation on NYUV2 dataset. Code is adapted from DenseDepth ([Code](https://github.com/ialhashim/DenseDepth))
## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the dataset from official sources and use simulation script to generate low-light SPC images.

## Training
```bash
python -u ../../../train.py --data-folder <Dataset folder> --bs 20 --epochs 20 --num-instance 5 --label-file <label-txt-file> --lamb 0.001 --pretrained-weights <imagenet-pretraining-weights> 2>&1 | tee train.txt
```

