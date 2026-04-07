import torch
import torch.nn as nn
from . import ConvWithNorms


class BilinearDecoder(nn.Module):

    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x,
                                         scale_factor=self.scale_factor,
                                         mode="bilinear",
                                         align_corners=False)

class UpsampleSkip(nn.Module):

    def __init__(self, skip_channels: int, latent_channels: int,
                 out_channels: int):
        super().__init__()
        self.u1_u2 = nn.Sequential(
            nn.Conv2d(skip_channels, latent_channels, 1, 1, 0),
            BilinearDecoder(2))
        self.u3 = nn.Conv2d(latent_channels, latent_channels, 1, 1, 0)
        self.u4_u5 = nn.Sequential(
            nn.Conv2d(2 * latent_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        u2_res = self.u1_u2(a)
        u3_res = self.u3(b)
        u5_res = self.u4_u5(torch.cat([u2_res, u3_res], dim=1))
        return u5_res


class FastFlow3DUNet(nn.Module):
    """
    Standard UNet with a few modifications:
     - Uses Bilinear interpolation instead of transposed convolutions
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder_step_1 = nn.Sequential(ConvWithNorms(32, 64, 3, 2, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1))
        self.encoder_step_2 = nn.Sequential(ConvWithNorms(64, 128, 3, 2, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1))
        self.encoder_step_3 = nn.Sequential(ConvWithNorms(128, 256, 3, 2, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1))
        self.decoder_step1 = UpsampleSkip(512, 256, 256)
        self.decoder_step2 = UpsampleSkip(256, 128, 128)
        self.decoder_step3 = UpsampleSkip(128, 64, 64)
        self.decoder_step4 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, pc0_B: torch.Tensor,
                pc1_B: torch.Tensor) -> torch.Tensor:

        expected_channels = 32
        assert pc0_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc0_B.shape[1]}"
        assert pc1_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc1_B.shape[1]}"

        pc0_F = self.encoder_step_1(pc0_B)
        pc0_L = self.encoder_step_2(pc0_F)
        pc0_R = self.encoder_step_3(pc0_L)

        pc1_F = self.encoder_step_1(pc1_B)
        pc1_L = self.encoder_step_2(pc1_F)
        pc1_R = self.encoder_step_3(pc1_L)

        Rstar = torch.cat([pc0_R, pc1_R],
                          dim=1)  # torch.Size([1, 512, 64, 64])
        Lstar = torch.cat([pc0_L, pc1_L],
                          dim=1)  # torch.Size([1, 256, 128, 128])
        Fstar = torch.cat([pc0_F, pc1_F],
                          dim=1)  # torch.Size([1, 128, 256, 256])
        Bstar = torch.cat([pc0_B, pc1_B],
                          dim=1)  # torch.Size([1, 64, 512, 512])

        S = self.decoder_step1(Rstar, Lstar)
        T = self.decoder_step2(S, Fstar)
        U = self.decoder_step3(T, Bstar)
        V = self.decoder_step4(U)

        return V

class UNetThreeFrame(nn.Module):
    """
    Standard UNet with a few modifications:
     - Uses Bilinear interpolation instead of transposed convolutions
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder_step_1 = nn.Sequential(ConvWithNorms(32, 64, 3, 2, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1),
                                            ConvWithNorms(64, 64, 3, 1, 1))
        self.encoder_step_2 = nn.Sequential(ConvWithNorms(64, 128, 3, 2, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1))
        self.encoder_step_3 = nn.Sequential(ConvWithNorms(128, 256, 3, 2, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1))
        self.decoder_step1 = UpsampleSkip(768, 384, 384)
        self.decoder_step2 = UpsampleSkip(384, 192, 192)
        self.decoder_step3 = UpsampleSkip(192, 96, 96)
        self.decoder_step4 = nn.Conv2d(96, 96, 3, 1, 1)

    def forward(self, pcb0_B: torch.Tensor, pc0_B: torch.Tensor,
                pc1_B: torch.Tensor) -> torch.Tensor:

        expected_channels = 32
        assert pc0_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc0_B.shape[1]}"
        assert pc1_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc1_B.shape[1]}"

        pcb0_F = self.encoder_step_1(pcb0_B)
        pcb0_L = self.encoder_step_2(pcb0_F)
        pcb0_R = self.encoder_step_3(pcb0_L)

        pc0_F = self.encoder_step_1(pc0_B)
        pc0_L = self.encoder_step_2(pc0_F)
        pc0_R = self.encoder_step_3(pc0_L)

        pc1_F = self.encoder_step_1(pc1_B)
        pc1_L = self.encoder_step_2(pc1_F)
        pc1_R = self.encoder_step_3(pc1_L)

        Rstar = torch.cat([pcb0_R, pc0_R, pc1_R],
                          dim=1)  # torch.Size([1, 512, 64, 64]), torch.Size([1, 768, 64, 64])  
        Lstar = torch.cat([pcb0_L, pc0_L, pc1_L],
                          dim=1)  # torch.Size([1, 256, 128, 128]), torch.Size([1, 384, 128, 128])
        Fstar = torch.cat([pcb0_F, pc0_F, pc1_F],
                          dim=1)  # torch.Size([1, 128, 256, 256]), torch.Size([1, 192, 256, 256])
        Bstar = torch.cat([pcb0_B, pc0_B, pc1_B],
                          dim=1)  # torch.Size([1, 64, 512, 512]), torch.Size([1, 96, 512, 512])

        S = self.decoder_step1(Rstar, Lstar)
        T = self.decoder_step2(S, Fstar)
        U = self.decoder_step3(T, Bstar)
        V = self.decoder_step4(U)

        return V

class ZeroFlowUNetXL(nn.Module):
    """
    Standard UNet with a few modifications:
     - Uses Bilinear interpolation instead of transposed convolutions
    """

    def __init__(self) -> None:
        super().__init__()

        self.encoder_step_1 = nn.Sequential(ConvWithNorms(64, 128, 3, 2, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1),
                                            ConvWithNorms(128, 128, 3, 1, 1))
        self.encoder_step_2 = nn.Sequential(ConvWithNorms(128, 256, 3, 2, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1),
                                            ConvWithNorms(256, 256, 3, 1, 1))
        self.encoder_step_3 = nn.Sequential(ConvWithNorms(256, 512, 3, 2, 1),
                                            ConvWithNorms(512, 512, 3, 1, 1),
                                            ConvWithNorms(512, 512, 3, 1, 1),
                                            ConvWithNorms(512, 512, 3, 1, 1),
                                            ConvWithNorms(512, 512, 3, 1, 1),
                                            ConvWithNorms(512, 512, 3, 1, 1))
        self.encoder_step_4 = nn.Sequential(ConvWithNorms(512, 1024, 3, 2, 1),
                                            ConvWithNorms(1024, 1024, 3, 1, 1),
                                            ConvWithNorms(1024, 1024, 3, 1, 1),
                                            ConvWithNorms(1024, 1024, 3, 1, 1),
                                            ConvWithNorms(1024, 1024, 3, 1, 1),
                                            ConvWithNorms(1024, 1024, 3, 1, 1))
        
        self.decoder_step1 = UpsampleSkip(2048, 1024, 1024)
        self.decoder_step2 = UpsampleSkip(1024, 512, 512)
        self.decoder_step3 = UpsampleSkip(512, 256, 256)
        self.decoder_step4 = UpsampleSkip(256, 128, 128)
        self.decoder_step5 = nn.Conv2d(128, 128, 3, 1, 1)

    def forward(self, pc0_B: torch.Tensor,
                pc1_B: torch.Tensor) -> torch.Tensor:

        expected_channels = 64
        assert pc0_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc0_B.shape[1]}"
        assert pc1_B.shape[
            1] == expected_channels, f"Expected {expected_channels} channels, got {pc1_B.shape[1]}"

        pc0_F = self.encoder_step_1(pc0_B)
        pc0_L = self.encoder_step_2(pc0_F)
        pc0_R = self.encoder_step_3(pc0_L)
        pc0_T = self.encoder_step_4(pc0_R)

        pc1_F = self.encoder_step_1(pc1_B)
        pc1_L = self.encoder_step_2(pc1_F)
        pc1_R = self.encoder_step_3(pc1_L)
        pc1_T = self.encoder_step_4(pc1_R)

        Tstar = torch.cat([pc0_T, pc1_T],
                          dim=1)  # torch.Size([1, 2048, 32, 32])
        Rstar = torch.cat([pc0_R, pc1_R],
                          dim=1)  # torch.Size([1, 1024, 64, 64])
        Lstar = torch.cat([pc0_L, pc1_L],
                          dim=1)  # torch.Size([1, 512, 128, 128])
        Fstar = torch.cat([pc0_F, pc1_F],
                          dim=1)  # torch.Size([1, 256, 256, 256])
        Bstar = torch.cat([pc0_B, pc1_B],
                          dim=1)  # torch.Size([1, 128, 512, 512])

        S = self.decoder_step1(Tstar, Rstar)
        T = self.decoder_step2(S, Lstar)
        U = self.decoder_step3(T, Fstar)
        V = self.decoder_step4(U, Bstar)
        W = self.decoder_step5(V)

        return W