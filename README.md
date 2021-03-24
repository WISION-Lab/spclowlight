## Photon-Starved Scene Inference using Single Photon Cameras
This repo implements the code for Photon Net


#### Code Structure
```bash
.
├── classification          # Code for image classification using Photon Net training
├── monodepth               # Code for monocular depth estimation using Photon Net training
├── simulation              # Scripts for simulating noisy SPAD images
└── README.md
```


## Requirements/Installation

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### How to Use
- Download the datasets (CUB/CARS/NYUV2/others) from the official sources and use scripts in `simulation` to simulate noisy images from SPAD 
- Use `classification` and `monodepth` code for image classifiation and monocular depth estimation using Photon Net 

