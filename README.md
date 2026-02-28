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
  <a href="https://correr-zhou.github.io/HiFi-Inpaint/"><img src="https://img.shields.io/static/v1?label=Paper&message=PDF&color=red"></a> &ensp;
  <a href="https://github.com/Correr-Zhou/HiFi-Inpaint"><img src="https://img.shields.io/static/v1?label=Code&message=GitHub&color=blue"></a> &ensp;
  <a href="https://huggingface.co/datasets/donghao-zhou/HP-Image-40K"><img src="https://img.shields.io/static/v1?label=Dataset&message=HP-Image-40K&color=yellow"></a> &ensp;
  <a href="https://github.com/Correr-Zhou/HiFi-Inpaint"><img src="https://img.shields.io/static/v1?label=Model&message=HiFi-Inpaint&color=yellow"></a>
</div>

---

## 🔥 Updates
- 2026.02: Our paper is accepted by CVPR 2026!

## 📑 Open-Source Plan
We will release the code, dataset, and model after internal review. Please stay tuned!
- [ ] HP-Image-40K Dataset
- [ ] HiFi-Inpaint Inference Code
- [ ] HiFi-Inpaint Model

## 🌍 Abstract
Human-product images, which showcase the integration of humans and products, play a vital role in advertising, e-commerce, and digital marketing. The essential challenge of generating such images lies in *ensuring the high-fidelity preservation of product details*. Among existing paradigms, *reference-based inpainting* offers a targeted solution by leveraging product reference images to guide the inpainting process. However, limitations remain in three key aspects: the lack of diverse large-scale training data, the struggle of current models to focus on product detail preservation, and the inability of coarse supervision for achieving precise guidance. To address these issues, we propose **HiFi-Inpaint**, a novel high-fidelity reference-based inpainting framework tailored for generating human-product images. HiFi-Inpaint introduces *Shared Enhancement Attention (SEA)* to refine fine-grained product features and *Detail-Aware Loss (DAL)* to enforce precise pixel-level supervision using high-frequency maps. Additionally, we construct a new dataset, *HP-Image-40K*, with samples curated from self-synthesis data and processed with automatic filtering. Experimental results show that HiFi-Inpaint achieves state-of-the-art performance, delivering detail-preserving human-product images.

<div align="center">
<img width="1080" alt="teaser" src="assets/teaser.jpg">
<p><small>We propose <b>HiFi-Inpaint</b>, a DiT-based framework that can seamlessly integrate product reference images into masked human images, generating high-quality human-product images with high-fidelity detail preservation.</small></p>
</div>

## 🔗 Citation
If you find HiFi-Inpaint useful for your research and applications, please cite:
```
@inproceedings{hifi_inpaint_2026,
  title={HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images},
  author={Liu, Yichen and Zhou, Donghao and Wang, Jie and Gao, Xin and Liu, Guisheng and Li, Jiatong and Zhang, Quanwei and Lyu, Qiang and Guo, Lanqing and Wen, Shilei and Wang, Weiqiang and Heng, Pheng-Ann},
  booktitle={CVPR},
  year={2026}
}
```
