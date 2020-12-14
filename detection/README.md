# Object detection using simulated images for SPCs

This contains instructions training models for object detection on simulated images of MS COCO dataset for SPCs using detectron2.
 
## Requirements

- Install PyTorch, [Detectron2](https://github.com/facebookresearch/detectron2)
- Download the dataset from official sources and use simulation script to generate low-light SPC images.
- Simulated images are in same directory structure and names as clean images (as expected by detectron2). Change the location in detectron folder to use simulated images.

## Training

To train a model, use `bfile` script .Make sure to update the dataset folder with simulated SPC images, mean and standard deviation values for the dataset (PIXEL\_MEAN and PIXEL\_STD in yaml file) and desired parameters. Refer to `outputC50/used.yaml` for example.

## Usage
```bash
./bfile
```

