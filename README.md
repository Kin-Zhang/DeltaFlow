DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method
---

[![arXiv](https://img.shields.io/badge/arXiv-2508.17054-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2508.17054)
[![poster](https://img.shields.io/badge/NeurIPS'25|Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte)](https://drive.google.com/file/d/1uh4brNIvyMsGLtoceiegJr-87K1wE_qo/view?usp=sharing)
[![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/YJ0HMZXnqxE)

<img width="1864" height="756" alt="deltaflow_cover" src="https://github.com/user-attachments/assets/a7348910-8073-4703-8c0b-57c613401552" />

**News w. TBD**:

Note (2025/09/18): We got accepted by NeurIPS 2025 and it's **spotlighted**! 🎉🎉🎉 The code are ready to play, enjoy!

- 2025/08/24: I'm updating some codes for early release. 
- [x] 2025/08/24: Updating train data augmentation as illustrated in the DeltaFlow paper.
- [x] 2025/08/25: Updating paper preprint link.
- [x] 2025/09/05: Merged the latest commit from OpenSceneFlow codebase to DeltaFlow for afterward unified merged.
- [x] 2025/09/25: DeltaFlow Model file, config file and loss function. Update quick training example.
- [x] 2025/09/29: Pre-trained weights for Argoverse 2, Waymo, nuScenes. _Contact me if any issue (e.g., ask for delete ckpt as privacy concern etc)._ These models are provided for research and reproducibility purposes only.
- [x] Public review comments for readers to refer to future improvement/directions etc. Refer discussion [here](https://github.com/Kin-Zhang/DeltaFlow/discussions/2).
- [ ] Merged into [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow), check pull request here: https://github.com/KTH-RPL/OpenSceneFlow/pull/21

## Quick Run

To train the full dataset, please refer to the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow?tab=readme-ov-file#1-data-preparation) for raw data download and h5py files preparation.

### Training

1. Prepare the **demo** train and val data for a quick run:
```bash
# around 1.3G
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo-data-v2.zip
unzip demo-data-v2.zip -d /home/kin/data/av2/h5py # to your data path
```

2. Follow the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow/tree/main?tab=readme-ov-file#0-installation) to setup the environment or [use docker](https://github.com/KTH-RPL/OpenSceneFlow?tab=readme-ov-file#docker-recommended-for-isolation).

3. Run the training with the following command (modify the data path accordingly):
```bash
python train.py model=deltaflow loss_fn=deltaflowLoss batch_size=4 num_frames=5 voxel_size="[0.15,0.15,0.15]" point_cloud_range="[-38.4,-38.4,-3,38.4,38.4,3]" optimizer.lr=2e-4 train_data=${demo_train_data_path} val_data=${demo_val_data_path}
```
### Evaluation

Here is the pretrained weights link table for different training datasets (Note that these models are only used for research and reproducibility purposes only please follow the dataset license and privacy rules to use them):

| Train Dataset | Pretrained ckpt Link |
|:--------:|:--------------:|
| Argoverse 2 | [huggingface](https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/deltaflow/deltaflow-av2.ckpt) |
| Waymo Open Dataset | [huggingface](https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/deltaflow/deltaflow-waymo.ckpt) |
| nuScenes | [huggingface](https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/deltaflow/deltaflow-nus.ckpt) |

Please check the local evaluation result (raw terminal output screenshot) in [this discussion thread](https://github.com/Kin-Zhang/DeltaFlow/discussions/1#discussion-8791273). 
You can also run the evaluation by yourself with the following command with trained weights:
```bash
python eval.py checkpoint=${path_to_pretrained_weights} dataset_path=${demo_data_path}
```

### Visualization

Please refer to the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow/tree/main?tab=readme-ov-file#4-visualization) for visualization instructions.

While I will update a unified visualization script for OpenSceneFlow to quickly save all window views as images at the same view and same time etc. (Free us from qualitative figure making work!)

## Cite & Acknowledgements
```
@inproceedings{zhang2025deltaflow,
title={{DeltaFlow}: An Efficient Multi-frame Scene Flow Estimation Method},
author={Zhang, Qingwen and Zhu, Xiaomeng and Zhang, Yushan and Cai, Yixi and Andersson, Olov and Jensfelt, Patric},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=T9qNDtvAJX}
}
```
This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation and Prosense (2020-02963) funded by Vinnova. 
