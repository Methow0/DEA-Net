# GLAND SEGMENTATION VIA DUAL ENCODERS AND BOUNDARY-ENHANCED ATTENTION (ICASSP 2024)

# Abstract
Accurate and automated gland segmentation on pathological images can assist pathologists in diagnosing the malignancy of colorectal adenocarcinoma. However, due to various gland shapes, severe deformation of malignant glands, and overlapping adhesions between glands. Gland segmentation has always been very challenging. To address these problems, we propose a DEA model. This model consists of two branches: the backbone encoding and decoding network and the local semantic extraction network. The backbone encoding and decoding network extracts advanced Semantic features, uses the proposed feature decoder to restore feature space information, and then enhances the boundary features of the gland through boundary enhancement attention. The local semantic extraction network uses the pre-trained DeepLabv3+ as a Local semantic-guided encoder to realize the extraction of edge features. Experimental results on two public datasets, GlaS and CRAG, conffrm that the performance of our method is better than other gland segmentation methods.

# Environment
- Ubuntu 20.04
- NVIDIA RTX 3090 GPU
- Python 3.8
- Pytorch 2.0.1

# Datasets
- [GlaS](https://datasets.activeloop.ai/docs/ml/datasets/glas-dataset/)
- [CRAG](https://warwick.ac.uk/fac/cross_fac/tia/data/mildnet/)

# Run the DCCL_Seg
python train.py --alpha=1 --batch-size=8 --lr=0.0005 --epochs=1000 --gpu=1 --save-dir=./experiments/GlaS/newcrop/DEA-Net

# Citation
If you find this code repository useful, please consider citing our paper:
```text
@INPROCEEDINGS{10447267,
  author={Wang, Huadeng and Yu, Jiejiang and Li, Bingbing and Pan, Xipeng and Liu, Zhenbing and Lan, Rushi and Luo, Xiaonan},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Gland Segmentation Via Dual Encoders and Boundary-Enhanced Attention}, 
  year={2024},
  volume={},
  number={},
  pages={2345-2349},
  keywords={Image segmentation;Semantics;Glands;Speech enhancement;Signal processing;Feature extraction;Encoding;Gland Segmentation;Dual Encoder;Local Semantic Guided Encoder;Boundary Enhance Attention},
  doi={10.1109/ICASSP48485.2024.10447267}}

```
