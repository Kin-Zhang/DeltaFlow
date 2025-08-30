
"""
# Created: 2023-07-18 15:08
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""

import torch.nn as nn
import dztimer, torch

from .basic.unet import FastFlow3DUNet, UNetThreeFrame
from .basic.encoder import DynamicEmbedder
from .basic.decoder import LinearDecoder, ConvGRUDecoder
from .basic import wrap_batch_pcs, BaseModel

class DeFlow(BaseModel):
    def __init__(self, voxel_size = [0.2, 0.2, 6],
                 point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3],
                 grid_feature_size = [512, 512],
                 decoder_option = "gru",
                 num_iters = 4):
        super().__init__()
        self.embedder = DynamicEmbedder(voxel_size=voxel_size,
                                        pseudo_image_dims=grid_feature_size,
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=32)
        
        self.backbone = FastFlow3DUNet()
        if decoder_option == "gru":
            self.head = ConvGRUDecoder(num_iters = num_iters)
        elif decoder_option == "linear":
            self.head = LinearDecoder()

        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def forward(self, batch):
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """
        self.timer[0].start("Data Preprocess")
        pcs_dict = wrap_batch_pcs(batch, num_frames=2)
        pc0s = pcs_dict['pc0s']
        pc1s = pcs_dict['pc1s']
        self.timer[0].stop()

        self.timer[1].start("Voxelization")
        pc0_before_pseudoimages, pc0_voxel_infos_lst = self.embedder(pc0s)
        pc1_before_pseudoimages, pc1_voxel_infos_lst = self.embedder(pc1s)
        self.timer[1].stop()

        self.timer[2].start("Encoder")
        grid_flow_pseudoimage = self.backbone(pc0_before_pseudoimages,
                                            pc1_before_pseudoimages)
        self.timer[2].stop()

        self.timer[3].start("Decoder")
        flows = self.head(
            torch.cat((pc0_before_pseudoimages, pc1_before_pseudoimages),
                    dim=1), grid_flow_pseudoimage, pc0_voxel_infos_lst)
        self.timer[3].stop()

        pc0_points_lst = [e["points"] for e in pc0_voxel_infos_lst]
        pc1_points_lst = [e["points"] for e in pc1_voxel_infos_lst]

        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_voxel_infos_lst]
        pc1_valid_point_idxes = [e["point_idxes"] for e in pc1_voxel_infos_lst]

        model_res = {
            "flow": flows,
            'pose_flow': pcs_dict['pose_flows'], 

            "pc0_valid_point_idxes": pc0_valid_point_idxes,
            "pc0_points_lst": pc0_points_lst,
            
            "pc1_valid_point_idxes": pc1_valid_point_idxes,
            "pc1_points_lst": pc1_points_lst,
            "num_occupied_voxels": [grid_flow_pseudoimage.size(-1)*grid_flow_pseudoimage.size(-2)]
        }
        return model_res



class DeFlowPP(BaseModel):
    def __init__(self, voxel_size = [0.2, 0.2, 6],
                 point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3],
                 grid_feature_size = [512, 512],
                 decoder_option = "gru",
                 num_iters = 2,
                 num_frames = 3):
        super().__init__()
        self.embedder = DynamicEmbedder(voxel_size=voxel_size,
                                        pseudo_image_dims=grid_feature_size,
                                        point_cloud_range=point_cloud_range,
                                        feat_channels=32)
        
        self.backbone = UNetThreeFrame()
        if decoder_option == "gru":
            self.head = ConvGRUDecoder(pseudoimage_channels=96, num_iters = num_iters)
        else:
            self.head = LinearDecoder()
        
        self.num_frames = num_frames
        assert self.num_frames == 3, "DeFlowPP only supports num_frames = 3"

        self.timer = dztimer.Timing()
        self.timer.start("Total")

    def forward(self, batch):
        """
        input: using the batch from dataloader, which is a dict
               Detail: [pc0, pc1, pose0, pose1]
        output: the predicted flow, pose_flow, and the valid point index of pc0
        """
        self.timer[0].start("Data Preprocess")
        pcs_dict = wrap_batch_pcs(batch, num_frames=self.num_frames)
        pc0s = pcs_dict['pc0s']
        pc1s = pcs_dict['pc1s']
        pch1s = pcs_dict['pch1s']
        self.timer[0].stop()

        self.timer[1].start("Voxelization")
        pch1_before_pseudoimages, pch1_voxel_infos_lst = self.embedder(pch1s)
        pc0_before_pseudoimages, pc0_voxel_infos_lst = self.embedder(pc0s)
        pc1_before_pseudoimages, pc1_voxel_infos_lst = self.embedder(pc1s)
        self.timer[1].stop()

        self.timer[2].start("Encoder")
        grid_flow_pseudoimage = self.backbone(pch1_before_pseudoimages, pc0_before_pseudoimages,
                                            pc1_before_pseudoimages)
        self.timer[2].stop()

        self.timer[3].start("Decoder")
        flows = self.head(
            torch.cat((pch1_before_pseudoimages, pc0_before_pseudoimages, pc1_before_pseudoimages),
                    dim=1), grid_flow_pseudoimage, pc0_voxel_infos_lst)
        self.timer[3].stop()

        model_res = {
            "flow": flows,
            'pose_flow': pcs_dict['pose_flows'],

            "pc0_valid_point_idxes": [e["point_idxes"] for e in pc0_voxel_infos_lst],
            "pc0_points_lst": [e["points"] for e in pc0_voxel_infos_lst],
            
            "pc1_valid_point_idxes":  [e["point_idxes"] for e in pc1_voxel_infos_lst],
            "pc1_points_lst": [e["points"] for e in pc1_voxel_infos_lst],

            'pch1_valid_point_idxes': [e["point_idxes"] for e in pch1_voxel_infos_lst],
            'pch1_points_lst': [e["points"] for e in pch1_voxel_infos_lst],
        }
        return model_res