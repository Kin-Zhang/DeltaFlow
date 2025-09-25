DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method
---

[![arXiv](https://img.shields.io/badge/arXiv-2508.17054-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2508.17054)
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seflow-a-self-supervised-scene-flow-method-in/self-supervised-scene-flow-estimation-on-1)](https://paperswithcode.com/sota/self-supervised-scene-flow-estimation-on-1?p=seflow-a-self-supervised-scene-flow-method-in) -->
<!-- [![poster](https://img.shields.io/badge/ECCV24|Poster-6495ed?style=flat&logo=Shotcut&logoColor=wihte)](https://hkustconnect-my.sharepoint.com/:b:/g/personal/qzhangcb_connect_ust_hk/EWyWD-tAX4xIma5U7ZQVk9cBVjsFv0Y_jAC2G7xAB-w4cg?e=c3FbMg)  -->
<!-- [![video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/fQqx2IES-VI) -->

**News w. TBD**:

Note (2025/09/18): We got accepted by NeurIPS 2025 and it's **spotlighted**! 🎉🎉🎉 Working on release the code here.

- 2025/08/24: I'm updating some codes for early release. 
- [x] 2025/08/24: Updating train data augmentation as illustrated in the DeltaFlow paper.
- [x] 2025/08/25: Updating paper preprint link.
- [x] 2025/09/05: Merged the latest commit from OpenSceneFlow codebase to DeltaFlow for afterward unified merged.
- [x] 2025/09/25: DeltaFlow Model file, config file and loss function. Update quick training example.
- [ ] pre-trained weights upload. Trying hard to find which one I used as tooo many ckpt files in my disk...
- [ ] Merged into [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow), check pull request here: https://github.com/KTH-RPL/OpenSceneFlow/pull/21

## Quick Run

### Training

1. Prepare the demo train and val data:
```bash
# around 1.3G
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo-data-v2.zip
unzip demo-data-v2.zip -d /home/kin/data/av2/h5py # to your data path
```

2. Follow the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow/tree/main?tab=readme-ov-file#0-installation) to setup the environment.

3. Run the training with the following command (modify the data path accordingly):
```bash
python train.py model=deltaflow loss_fn=deltaflowLoss batch_size=4 num_frames=5 voxel_size="[0.15,0.15,0.15]" point_cloud_range="[-38.4,-38.4,-3,38.4,38.4,3]" optimizer.lr=2e-4 train_data=${demo_train_data_path} val_data=${demo_val_data_path}
```
### Evaluation

I will provide the pre-trained weights soon. Then you can run the evaluation with the following command:
```bash
python eval.py checkpoint=${path_to_pretrained_weights} dataset_path=${demo_data_path}
```

### Visualization

Please refer to the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow/tree/main?tab=readme-ov-file#4-visualization) for visualization instructions.

While I will update a unified visualization script for OpenSceneFlow to quickly save all window views as images at the same view and same time etc. (Free us from qualitative figure making work!)



## Cite & Acknowledgements
```
@article{zhang2025deltaflow,
    title={{DeltaFlow}: An Efficient Multi-frame Scene Flow Estimation Method},
    author={Zhang, Qingwen and Zhu, Xiaomeng and Zhang, Yushan and Cai, Yixi and Andersson, Olov and Jensfelt, Patric},
    year={2025},
    journal={arXiv preprint arXiv:2508.17054},
}
```
This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation and Prosense (2020-02963) funded by Vinnova. 
