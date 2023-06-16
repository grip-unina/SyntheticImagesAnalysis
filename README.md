# Intriguing properties of synthetic images: from generative adversarial networks to diffusion models

[![Github](https://img.shields.io/badge/Github%20webpage-222222.svg?style=for-the-badge&logo=github)](https://grip-unina.github.io/SyntheticImagesAnalysis/)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://doi.org/10.48550/arXiv.2304.06408)
[![CVF](https://img.shields.io/badge/-CVF-6093BF.svg?style=for-the-badge)](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Corvi_Intriguing_Properties_of_Synthetic_Images_From_Generative_Adversarial_Networks_to_CVPRW_2023_paper.html)
[![GRIP](https://img.shields.io/badge/-GRIP-0888ef.svg?style=for-the-badge)](https://www.grip.unina.it)



## (Code Coming Soon)
Official implementation of the paper: "Intriguing properties of synthetic images: from generative adversarial networks to diffusion models". 

<p align="center">
 <img src="./docs/Preview.png" alt="Preview" width="95%" />
</p>




## Overview

Detecting fake images is becoming a major goal of computer vision. This need is becoming more and more pressing with the continuous improvement of synthesis methods based on Generative Adversarial Networks (GAN), and even more with the appearance of powerful methods based on Diffusion Models (DM). Towards this end, it is important to gain insight into which image features better discriminate fake images from real ones. In this paper we report on our systematic study of a large number of image generators of different families, aimed at discovering the most forensically relevant characteristics of real and generated images. Our experiments provide a number of interesting observations and shed light on some intriguing properties of synthetic images: (1) not only GANs architectures give rise to artifacts visible in the Fourier domain, but also DMs and VQ-GANs (Vector Quantized Generative Adversarial Networks) present irregular patterns; (2) when the dataset used to train the model lacks sufficient variety, its biases can be transferred to the generated images; (3) synthetic and real images differ statistically in the mid-high frequency signal content, observable in their radial and angular spectral energy distribution.

## License

## Bibtex 

```
@inproceedings{corvi2023intriguing,
  title={Intriguing properties of synthetic images: from generative adversarial networks to diffusion models},
  author={Corvi, Riccardo and Cozzolino, Davide and Poggi, Giovanni and Nagano, Koki and Verdoliva, Luisa},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={973--982},
  year={2023}
}
```
