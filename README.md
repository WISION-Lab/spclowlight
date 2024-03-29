## Photon-Starved Scene Inference using Single Photon Cameras

**ICCV 2021** <br> [Arxiv](https://arxiv.org/abs/2107.11001) &nbsp; [Project](https://wisionlab.cs.wisc.edu/project/photon-net/) &nbsp; [Video](https://www.youtube.com/watch?v=r1YvHnGbi6k)

![teaser](figures/ArchitectureVersatile-0.png)

#### [Bhavya Goyal](https://bhavyagoyal.github.io), [Mohit Gupta](https://wisionlab.cs.wisc.edu/people/mohit-gupta/)
University of Wisconsin-Madison

### Abstract
Scene understanding under low-light conditions is a challenging problem. This is due to the small number of photons captured by the camera and the resulting low signal-to-noise ratio (SNR). Single-photon cameras (SPCs) are an emerging sensing modality that are capable of cap-turing images with high sensitivity. Despite having minimal read-noise, images captured by SPCs in photon-starved conditions still suffer from strong shot noise, preventing reliable scene inference. We propose photon scale-space, a collection of high-SNR images spanning a wide range of photons-per-pixel (PPP) levels (but same scene content) as guides to train inference model on low photon flux images. We develop training techniques that push images with different illumination levels closer to each other in feature representation space. The key idea is that having a spectrum of different brightness levels during training enables effective guidance, and increases robustness to shot noise even in extreme noise cases. Based on the proposed approach, we demonstrate, via simulations and real experiments with a SPAD camera, high-performance on various inference tasks such as image classification and monocular depth estimation under ultra low-light, down to < 1 PPP.

### Code Structure
```bash
.
├── classification          # Code for image classification using Photon Net training
├── monodepth               # Code for monocular depth estimation using Photon Net training
├── simulation              # Scripts for simulating noisy SPAD images
├── figures                 # figures used for results
└── README.md
```


### Requirements/Installation

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### How to Use
- Download the datasets (CUB/CARS/NYUV2/others) from the official sources and use scripts in `simulation` to simulate noisy images from SPAD
- Use `classification` and `monodepth` code for image classifiation and monocular depth estimation using Photon Net


### Citation
```
@InProceedings{Goyal_2021_ICCV,
    author    = {Goyal, Bhavya and Gupta, Mohit},
    title     = {Photon-Starved Scene Inference Using Single Photon Cameras},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2512-2521}
}
```
