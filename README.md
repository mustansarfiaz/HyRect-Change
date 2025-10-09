# HYRET-CHANGE: A HYBRID RETENTIVE NETWORK FOR REMOTE SENSING CHANGE DETECTION

#### [Mustansar Fiaz](https://sites.google.com/view/mustansarfiaz/home), [Mubashir Noman](https://scholar.google.com/citations?user=S6_CVskAAAAJ&hl=en),  [Hiyam Debary](https://www.linkedin.com/in/hiyam-debary/), [Kamran Ali](https://scholar.google.com/citations?user=JuQ_vNIAAAAJ&hl=en), [Hisham Cholakkal](https://hishamcholakkal.com/)


[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2506.12836)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-FF9900)](https://huggingface.co/mustansarfiaz/HyRet)


---

### 🏆 Highlights
-----------------
- **HyRet-Change:** We propose a Siamese-based framework, which can seamlessly integrate the merits of convolution and retention mechanisms at multi-scale features to preserve critical information and enhance adaptability in complex scenes change detection (CD).  Specifically, we propose a hybrid plug-and-play feature difference module (FDM) to explore rich feature information utilizing both self-attention and convolution operations in a parallel way. This unique integration, at multi-scale features, leverages the
advantages of both local features and long-range contextual information. We introduce a retention mechanism in our novel FDM to mitigate the limitations of standard self-attention.
- **Local-Global (LG)-Interaction Module:** We introduce an adaptive interaction between local and global representations to exploit the intricate relationship contextually to strengthen the model’s ability to perceive meaningful changes while reducing the effect of pseudo-changes.
- **Experiments:** Our extensive experimental study over three challenging CD datasets demonstrates the merits of our approach while achieving state-of-the-art performance.

---
### 👁️💬 Proposed Framework
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/HyRect-Change/blob/main/images/fig1.png">

---
###  📊 Quantitative Comparison
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/HyRect-Change/blob/main/images/table1.png">

---
### :speech_balloon: Qualitative Comparison from the LEVIR-CD (first row) and WHU-CD (second row) datasets
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/HyRect-Change/blob/main/images/qualitative.png">

---

### Requirements
```
Python 3.8.0
pytorch 1.10.1
torchvision 0.11.2
einops  0.3.2
```

---
### :speech_balloon: Dataset Preparation

### :point_right: Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

---
## Citation

```
@inproceedings{fiaz2025hyret,
  title={HyRet-Change: A hybrid retentive network for remote sensing change detection},
  author={Fiaz, Mustansar and Noman, Mubashir and Debary, Hiyam and Ali, Kamran and Cholakkal, Hisham},
  booktitle={IGARSS 2025-2025 IEEE International Geoscience and Remote Sensing Symposium},
  year={2025},
  publisher={IEEE}
}
@inproceedings{noman2024changebind,
  title={Changebind: A hybrid change encoder for remote sensing change detection},
  author={Noman, Mubahsir and Fiaz, Mustansar and Cholakkal, Hisham},
  booktitle={IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium},
  pages={8417--8422},
  year={2024},
  organization={IEEE}
}
@article{noman2024elgc,
  title={ELGC-Net: Efficient local--global context aggregation for remote sensing change detection},
  author={Noman, Mubashir and Fiaz, Mustansar and Cholakkal, Hisham and Khan, Salman and Khan, Fahad Shahbaz},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--11},
  year={2024},
  publisher={IEEE}
}
@article{noman2024remote,
  title={Remote sensing change detection with transformers trained from scratch},
  author={Noman, Mubashir and Fiaz, Mustansar and Cholakkal, Hisham and Narayan, Sanath and Anwer, Rao Muhammad and Khan, Salman and Khan, Fahad Shahbaz},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--14},
  year={2024},
  publisher={IEEE}
}
```

### Contact

If you have any question, please feel free to contact the authors. Mustansar Fiaz: [mustansar.fiaz@ibm.com](mailto:mustansar.fiaz@ibm.com) or Mubashir Noman: [mubashir.noman@mbzuai.ac.ae](mailto:mubashir.noman@mbzuai.ac.ae).

## References
Our code is based on [Changebind](https://github.com/techmn/changebind) repository. 
We thank them for releasing their baseline code.

