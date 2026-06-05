<h1 align="center" style="line-height: 50px;">
  HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images
</h1>

<div align="center">
Yichen Liu<sup>1,*</sup>, Donghao Zhou<sup>2,*</sup>, Jie Wang<sup>3</sup>, Xin Gao<sup>3</sup>, Guisheng Liu<sup>3</sup>, Jiatong Li<sup>3,†</sup>, Quanwei Zhang<sup>4</sup>,<br>
Qiang Lyu<sup>1</sup>, Lanqing Guo<sup>5</sup>, Shilei Wen<sup>3,§</sup>, Weiqiang Wang<sup>1,§</sup>, Pheng-Ann Heng<sup>2,§</sup>
</div>

<br>

<div align="center">
<sup>1</sup>University of Chinese Academy of Sciences, <sup>2</sup>The Chinese University of Hong Kong, <sup>3</sup>ByteDance,<br>
<sup>4</sup>Zhejiang University, <sup>5</sup>UT Austin
</div>

<br>

<div align="center">
*Equal contribution, †Project Lead, §Corresponding Author
</div>

<br>

<div align="center">
  <a href="https://correr-zhou.github.io/HiFi-Inpaint/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/pdf/2603.02210"><img src="https://img.shields.io/static/v1?label=Paper&message=PDF&color=red"></a> &ensp;
  <a href="https://github.com/Correr-Zhou/HiFi-Inpaint"><img src="https://img.shields.io/static/v1?label=Code&message=GitHub&color=blue"></a> &ensp;
  <a href="https://huggingface.co/datasets/donghao-zhou/HP-Image-40K"><img src="https://img.shields.io/static/v1?label=Dataset&message=HP-Image-40K&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/datasets/donghao-zhou/HP-Image-40K"><img src="https://img.shields.io/static/v1?label=Model&message=HiFi-Inpaint&color=yellow"></a>
</div>

---

## 🔥 Updates
- 2026.06: Inference code, training code, and model weights are released!
- 2026.02: Our paper is accepted by CVPR 2026!

## ✅ Open-Source Plan
- [x] HP-Image-40K Dataset
- [x] HiFi-Inpaint Inference Code
- [x] HiFi-Inpaint Training Code
- [x] HiFi-Inpaint Model Weights

## 🌍 Abstract
Human-product images, which showcase the integration of humans and products, play a vital role in advertising, e-commerce, and digital marketing. The essential challenge of generating such images lies in *ensuring the high-fidelity preservation of product details*. Among existing paradigms, *reference-based inpainting* offers a targeted solution by leveraging product reference images to guide the inpainting process. However, limitations remain in three key aspects: the lack of diverse large-scale training data, the struggle of current models to focus on product detail preservation, and the inability of coarse supervision for achieving precise guidance. To address these issues, we propose **HiFi-Inpaint**, a novel high-fidelity reference-based inpainting framework tailored for generating human-product images. HiFi-Inpaint introduces *Shared Enhancement Attention (SEA)* to refine fine-grained product features and *Detail-Aware Loss (DAL)* to enforce precise pixel-level supervision using high-frequency maps. Additionally, we construct a new dataset, *HP-Image-40K*, with samples curated from self-synthesis data and processed with automatic filtering. Experimental results show that HiFi-Inpaint achieves state-of-the-art performance, delivering detail-preserving human-product images.

<div align="center">
<img width="1080" alt="teaser" src="assets/teaser.jpg">
<p><small>We propose <b>HiFi-Inpaint</b>, a DiT-based framework that can seamlessly integrate product reference images into masked human images, generating high-quality human-product images with high-fidelity detail preservation.</small></p>
</div>

## 🛠️ Environment Setup

We recommend using a clean Conda environment with Python 3.11:

```bash
git clone https://github.com/Correr-Zhou/HiFi-Inpaint.git
cd HiFi-Inpaint

conda create -n hifi-inpaint python=3.11 -y
conda activate hifi-inpaint

pip install -r requirements.txt
```

If the default PyTorch installation does not match your CUDA version, reinstall PyTorch manually. For example, for CUDA 12.4:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

## ⚡ Inference

1. Download the base model [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and our [LoRA weights](https://huggingface.co/datasets/donghao-zhou/HP-Image-40K).

2. Update the paths in `run_inference.sh`:
   - `FLUX_PATH`: path to FLUX.1-dev
   - `LORA_PATH`: path to LoRA checkpoint

3. Run inference:
```bash
bash run_inference.sh
```

Results will be saved to `./output/`.

## 📦 Training

Our training data format is compatible with [HP-Image-40K](https://huggingface.co/datasets/donghao-zhou/HP-Image-40K). Each JSON entry should contain:

```json
{
    "ref_image_path": "path/to/reference_image.png",
    "gt_image_path": "path/to/ground_truth_image.png",
    "condition_image_path": "path/to/masked_condition_image.png",
    "mask_path": "path/to/binary_mask.png",
    "caption": "text description of the image"
}
```

1. Update the config file `train/config/hifi_inpaint.yaml`:
   - `flux_path`: path to [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
   - `data_path`: path to your training data JSON file(s)

2. Run training:

```bash
# Single GPU
bash train/scripts/train.sh

# Multi-GPU with torchrun
torchrun --nproc_per_node=8 -m src.train.train
```

Training logs are tracked via [Weights & Biases](https://wandb.ai/). Set your API key before training:
```bash
export WANDB_API_KEY="your_wandb_api_key"
```

See `train/config/` for more training configurations.

## 🤝 Acknowledgements

Our codebase is built upon [OminiControl](https://github.com/Yuanshi9815/OminiControl). We thank the authors for their excellent work.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Citation
If you find HiFi-Inpaint useful for your research and applications, please cite:
```
@article{liu2026hifiinpaint,
  title={HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images},
  author={Liu, Yichen and Zhou, Donghao and Wang, Jie and Gao, Xin and Liu, Guisheng and Li, Jiatong and Zhang, Quanwei and Lyu, Qiang and Guo, Lanqing and Wen, Shilei and Wang, Weiqiang and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2603.02210},
  year={2026}
}
```
