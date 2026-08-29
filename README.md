# Image Watermarks are Removable using Controllable Regeneration from Clean Noise

Official implementation of an **invisible image watermark removing** method presented in the paper:

["Image Watermarks Are Removable using Controllable Regeneration from Clean Noise"](https://arxiv.org/abs/2410.05470) by Yepeng Liu, Yiren Song, Hai Ci, Yu Zhang, Haofan Wang, Mike Zheng Shou, and Yuheng Bu.

## Introduction

Image watermark techniques provide an effective way to assert ownership, deter misuse, and trace content sources, which has become increasingly essential in the era of large generative models. A critical attribute of watermark techniques is their robustness against various manipulations. In this paper, we introduce a watermark removal approach capable of effectively nullifying the state-of-the-art watermarking techniques. Our primary insight involves regenerating the watermarked image starting from a **clean Gaussian noise** via a controllable diffusion model, utilizing the extracted semantic and spatial features from the watermarked image. The semantic control adapter and the spatial control network are specifically trained to control the denoising process towards ensuring image quality and enhancing consistency between the cleaned image and the original watermarked image. To achieve a smooth trade-off between watermark removal performance and image consistency, we further propose an adjustable and controllable regeneration scheme. This scheme adds varying numbers of noise steps to the latent representation of the watermarked image, followed by a controlled denoising process starting from this noisy latent representation. As the number of noise steps increases, the latent representation progressively approaches clean Gaussian noise, facilitating the desired trade-off. We apply our watermark removal methods across various watermarking techniques, and the results demonstrate that our methods offer superior visual consistency/quality and enhanced watermark removal performance compared to existing regeneration approaches.


![overview](https://github.com/user-attachments/assets/0776782c-e5d7-4fea-9600-c3ec6f58c812)


## Installation
```
# download the code
git clone https://github.com/yepengliu/CtrlRegen.git
cd CtrlRegen

# install the required environment
pip install -r requirements.txt
```

## How to use
The checkpoints for the Semantic Control Adapter and Spatial Control Network are available on `https://huggingface.co/yepengliu/ctrlregen` for direct use; feel free to download them or train your own models as needed.

Please see [**CtrlRegen_Plus_Demo**](ctrlregen_plus_demo.ipynb): Adjust the watermark removal strength by changing the step parameter between 0 and 1.

Below are some examples. The left one represents the original image, while the right one shows the regenerated image using our method (step = 0.5).

<p><img src="https://github.com/user-attachments/assets/7a112970-d60d-4760-80d9-a4f42f905526" width="500"></p>

<p><img src="https://github.com/user-attachments/assets/1dcbbdc5-d2f4-4de6-b064-b5ffc1a45ccd" width="500"></p>

<p><img src="https://github.com/user-attachments/assets/768a0fcb-ba72-4242-bf0e-579c11b3558a" width="500"></p>

<p><img src="https://github.com/user-attachments/assets/e83a07da-0a74-40e4-aa43-d256f7fdb3ad" width="500"></p>

<p><img src="https://github.com/user-attachments/assets/93295f22-e264-48ce-afc7-35c59b3256e7" width="500"></p>

## How to train

Training code is in [`train/`](train). CtrlRegen is trained in two stages:

1. **Semantic Control** (`train_semantic.py`)
2. **Spatial Control** (`train_spatial.py`)

`convert_checkpoint.py` turns the resulting checkpoints into the exact weight format this repo's
inference notebook loads.

See [`train/README.md`](train/README.md) for setup, data format, hyperparameters and the
configuration used for the released weights.

## Disclaimer
This project aims to enhance the robustness of image watermarks. Users are free to leverage this tool to strengthen their watermarking algorithms; however, they must **adhere to local regulations and use it responsibly**. The developers disclaim any liability for potential misuse by users.

## Citation
If you find this repository useful for your research or applications, please cite our paper:
```
@article{liu2024ctrlregen,
  title={Image watermarks are removable using controllable regeneration from clean noise},
  author={Liu, Yepeng and Song, Yiren and Ci, Hai and Zhang, Yu and Wang, Haofan and Shou, Mike Zheng and Bu, Yuheng},
  journal={arXiv preprint arXiv:2410.05470},
  year={2024}
}
```

## Acknowledgement
We thank the authors of the following research works and open-source projects:

["IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models"](https://github.com/tencent-ailab/IP-Adapter)

["Adding Conditional Control to Text-to-Image Diffusion Models"](https://github.com/lllyasviel/ControlNet?tab=readme-ov-file)

