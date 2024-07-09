## Table of Contents

- [Table of Contents](#table-of-contents)
- [License](#license)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Checkpoints](#checkpoints)
  - [usage.py](#usagepy)
  - [Interactive demo](#interactive-demo)

## License

The code is distributed under the CC BY-NC-SA 4.0 License.
See [LICENSE](LICENSE) for more information.

```python
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
```

## Get Started

### Installation

1. Clone DUSt3R.
```bash
git clone --recursive https://github.com/naver/dust3r
cd dust3r
# if you have already cloned dust3r:
# git submodule update --init --recursive
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
pip install -r requirements_optional.txt
```

3. Optional, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### Checkpoints

You can obtain the checkpoints by two ways:

1) You can use our huggingface_hub integration: the models will be downloaded automatically.

2) Otherwise, We provide several pre-trained models:

| Modelname   | Training resolutions | Head | Encoder | Decoder |
|-------------|----------------------|------|---------|---------|
| [`DUSt3R_ViTLarge_BaseDecoder_224_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth) | 224x224 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_linear.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth)   | 512x384, 512x336, 512x288, 512x256, 512x160 | Linear | ViT-L | ViT-B |
| [`DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth) | 512x384, 512x336, 512x288, 512x256, 512x160 | DPT | ViT-L | ViT-B |

You can check the hyperparameters we used to train these models in the [section: Our Hyperparameters](#our-hyperparameters)

To download a specific model, for example `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`:
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```
### usage.py

This is our modified script. You can run this script by following:

```bash
python usage.py

# Currently, you need to hard-code the model and image paths. An argument parser will be integrated in the future.
# Please adjust the model_name and image_file_ls variables to match the respective file locations.
```

### Interactive demo

In this demo, you should be able run DUSt3R on your machine to reconstruct a scene.
First select images that depicts the same scene.

You can adjust the global alignment schedule and its number of iterations.

> [!NOTE]
> If you selected one or two images, the global alignment procedure will be skipped (mode=GlobalAlignerMode.PairViewer)

Hit "Run" and wait.
When the global alignment ends, the reconstruction appears.
Use the slider "min_conf_thr" to show or remove low confidence areas.

```bash
python3 demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt

# Use --weights to load a checkpoint from a local file, eg --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
# Use --image_size to select the correct resolution for the selected checkpoint. 512 (default) or 224
# Use --local_network to make it accessible on the local network, or --server_name to specify the url manually
# Use --server_port to change the port, by default it will search for an available port starting at 7860
# Use --device to use a different device, by default it's "cuda"
```

