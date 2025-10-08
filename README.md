# HYRET-CHANGE: A HYBRID RETENTIVE NETWORK FOR REMOTE SENSING CHANGE DETECTION

This repo contains the official **PyTorch** code for HYRET-CHANGE: A HYBRID RETENTIVE NETWORK FOR REMOTE SENSING CHANGE DETECTION [[Arxiv]](https://arxiv.org/pdf/2506.12836). 

**Code is released!**

Highlights
-----------------
- **HyRet-Change:** We propose a Siamese-based framework, which can seamlessly integrate the merits of convolution and retention mechanisms at multi-scale features to preserve critical information and enhance adaptability in complex scenes change detection (CD).  Specifically, we propose a hybrid plug-and-play feature difference module (FDM) to explore rich feature information utilizing both self-attention and convolution operations in a parallel way. This unique integration, at multi-scale features, leverages the
advantages of both local features and long-range contextual information. We introduce a retention mechanism in our novel FDM to mitigate the limitations of standard self-attention.
- **Local-Global (LG)-Interaction Module:** We introduce an adaptive interaction between local and global representations to exploit the intricate relationship contextually to strengthen the model’s ability to perceive meaningful changes while reducing the effect of pseudo-changes.
- **Experiments:** Our extensive experimental study over three challenging CD datasets demonstrates the merits of our approach while achieving state-of-the-art performance.

### :speech_balloon: Proposed Framework
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/HyRect-Change/blob/main/images/fig1.png">

### :speech_balloon: Quantitative Comparison
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/HyRect-Change/blob/main/images/table1.png">

### :speech_balloon: Qualitative Comparison from the LEVIR-CD (first row) and WHU-CD (second row) datasets
-----------------
<img width="1096" alt="image" src="https://github.com/mustansarfiaz/HyRect-Change/blob/main/images/qualitative.png">


### Requirements
```
Python 3.8.0
pytorch 1.10.1
torchvision 0.11.2
einops  0.3.2
```

Please see `requirements.txt` for all the other requirements.

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


## Citation

```
@article{fiaz2025hyret,
  title={HyRet-Change: A hybrid retentive network for remote sensing change detection},
  author={Fiaz, Mustansar and Noman, Mubashir and Debary, Hiyam and Ali, Kamran and Cholakkal, Hisham},
  journal={arXiv preprint arXiv:2506.12836},
  year={2025}
}
@misc{changebind2024,
  title={ChangeBind: A Hybrid Change Encoder for Remote Sensing Change Detection}, 
  author={Mubashir Noman and Mustansar Fiaz and Hisham Cholakkal},
  year={2024},
  eprint={2404.17565},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
  url={https://arxiv.org/abs/2404.17565}, 
}
@article{noman2024elgc,
  title={ELGC-Net: Efficient Local-Global Context Aggregation for Remote Sensing Change Detection},
  author={Noman, Mubashir and Fiaz, Mustansar and Cholakkal, Hisham and Khan, Salman and Khan, Fahad Shahbaz},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
@inproceedings{noman2023scratchformer,
  title={Remote Sensing Change Detection with Transformers Trained from Scratch},
  author={Noman, Mubashir and Fiaz, Mustansar and Cholakkal, Hisham and Narayan, Sanath and Anwer, Rao Muhammad and Khan, Salman and Khan, Fahad Shahbaz},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024}
}
```

### Contact

If you have any question, please feel free to contact the authors. Mustansar Fiaz: [mustansar.fiaz@mbzuai.ac.ae](mailto:mustansar.fiaz@mbzuai.ac.ae) or Mubashir Noman: [mubashir.noman@mbzuai.ac.ae](mailto:mubashir.noman@mbzuai.ac.ae).

## References
Our code is based on [Changebind](https://github.com/techmn/changebind) repository. 
We thank them for releasing their baseline code.

