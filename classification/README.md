# Image Classification on noisy images from SPADs

This implements training of image classification using Photon Net on noisy images from spads. Modified from pytorch examples
## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the dataset from official sources and use simulation script to generate low-light SPC images.

## Training
```bash
python -u ./main.py <dataset_folder>  --cub-training  --use-photon-net  <pretrained_weights_on_clean_images_location>\
 --experiment=photon_net_run1 2>&1 --epochs 100 --b 80 --workers 8\
 --mean $MEANINPUT  --lamb 0.5 --num-instances 5 --label-file <label_txt_file> | tee train.txt
```


