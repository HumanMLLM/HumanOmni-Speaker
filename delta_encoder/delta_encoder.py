"""Visual Delta Encoder: SVT 结构化视觉 tokenizer + Transformer 全局编码 + 投影头."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder

from .resnet import ResNetVideoFrontend, ResidualBlock1D, ResidualBlock3DSplit


# =====================================================================
# SVT: Structured Visual Tokenizer
# =====================================================================

def create_separable_pos_conv(e, k, g):
    spatial_conv = nn.Conv3d(
        in_channels=e,
        out_channels=e,
        kernel_size=(1, 3, 3),
        stride=(1, 1, 1),
        padding=(0, 1, 1),
        groups=g,
        bias=False
    )
    temporal_conv = nn.Conv3d(
        in_channels=e,
        out_channels=e,
        kernel_size=(k, 1, 1),
        stride=(1, 1, 1),
        padding=(k // 2, 0, 0),
        groups=g,
        bias=False
    )

    std = math.sqrt(4.0 / (k * e))
    nn.init.normal_(spatial_conv.weight, mean=0, std=std)
    nn.init.normal_(temporal_conv.weight, mean=0, std=std)

    pos_conv = nn.Sequential(
        nn.BatchNorm3d(e),
        spatial_conv,
        nn.GELU(),
        nn.BatchNorm3d(e),
        temporal_conv,
        nn.GELU()
    )
    return pos_conv


class StructuredVisualTokenizer(nn.Module):
    """SVT (Structured Visual Tokenizer): 把每帧 CNN 特征压缩成 6 个结构化 token.

    7x7 空间切分 (patchify_conv) + k=63 大感受野时空位置卷积 (spatiotemporal_pos_conv),
    structured_token_emb 的 6 个槽位即每帧的 6 个结构化 token。
    """
    def __init__(self, in_channels, embed_dim):
        super(StructuredVisualTokenizer, self).__init__()
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.patch_size = 7
        self.patchify_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.spatiotemporal_pos_conv = create_separable_pos_conv(embed_dim, 63, 16)

        self.structured_token_emb = nn.Parameter(torch.randn(1, 1, 6, embed_dim))  # B,T,HW,D

    def forward(self, x):
        B, Tnew, channel, Height, Width = x.shape

        x = x.view(B * Tnew, channel, Height, Width)

        if Height * Width <= 1.3 * self.patch_size * self.patch_size:
            x = F.interpolate(x, size=(self.patch_size, self.patch_size), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=(2 * self.patch_size, 3 * self.patch_size), mode='bilinear', align_corners=True)

        x = self.patchify_conv(x)

        new_channel = x.shape[1]

        x = x.view(B, Tnew, *x.shape[1:])

        x_conv = self.spatiotemporal_pos_conv(x.transpose(1, 2))
        x = x + x_conv.transpose(1, 2)

        x = x.view(B * Tnew, new_channel, -1).transpose(1, 2).contiguous()  # [B*Tnew, H*W, D]
        hw = x.shape[1]
        x = x.view(B, Tnew, hw, new_channel).contiguous()

        x = x.view(B, Tnew * hw, new_channel).contiguous()

        x = self.ln_post(self.proj(x))

        if hw > 1:
            slot_emb = self.structured_token_emb.expand(B, Tnew, -1, -1).reshape(B, Tnew * hw, new_channel)
            x = (x + slot_emb).to(x.dtype)
        else:
            random_index = torch.randint(0, 6, (1,)).item()
            slot_emb = self.structured_token_emb[:, :, random_index, :]
            slot_emb = slot_emb.expand(B, Tnew, -1)
            x = (x + slot_emb).to(x.dtype)

        return x


class DeltaStream25fps(nn.Module):
    """Visual Delta Encoder 的视觉流: ResNet-18 局部感知 + SVT 结构化 tokenize."""
    def __init__(self, resnet=None, input_dim=None, encoder_embed_dim=None, av_corr=False):
        super().__init__()
        self.av_corr = av_corr
        self.resnet = resnet
        self.svt = StructuredVisualTokenizer(in_channels=input_dim, embed_dim=encoder_embed_dim)
        self.video_backend = nn.Sequential(
            ResidualBlock3DSplit(input_dim, input_dim),
            ResidualBlock3DSplit(input_dim, input_dim),
        )
        self.audio_backend = nn.Sequential(
            ResidualBlock1D(input_dim, input_dim),
            ResidualBlock1D(input_dim, input_dim),
        )

    def forward(self, videos, audios=None, video_lengths=None):
        if self.resnet is not None:
            videos = self.resnet(videos)

        if audios is not None and video_lengths is not None and self.av_corr:
            heatmaps = []
            seq_len = videos.shape[1]

            for idx, video_length in enumerate(video_lengths):
                audio = audios[idx:idx + 1, :video_length, :]
                video = videos[idx:idx + 1, :video_length, :, :, :]
                audio = self.audio_backend(audio.transpose(1, 2)).transpose(1, 2).contiguous()
                video = video.transpose(1, 2).contiguous()
                video = self.video_backend(video).transpose(1, 2).contiguous()
                heatmap = F.relu(F.cosine_similarity(audio.unsqueeze(-1).unsqueeze(-1), video, dim=2))

                heatmap = heatmap.unsqueeze(2)
                padding_size = seq_len - heatmap.shape[1]
                if padding_size > 0:
                    heatmap = F.pad(heatmap, (0, 0, 0, 0, 0, 0, 0, padding_size))
                heatmaps.append(heatmap)
            heatmap = torch.cat(heatmaps)
            videos = heatmap * videos

        videos = self.svt(videos)

        return videos


class DeltaEncoderBackbone(nn.Module):
    """Visual Delta Encoder: Local Feature Perception (ResNet-18) + SVT + Global Context Encoding (Transformer)."""
    def __init__(self, cfg):
        super(DeltaEncoderBackbone, self).__init__()

        resnet = ResNetVideoFrontend(relu_type='prelu')

        self.skip_pos_conv = True
        self.delta_stream_25fps = DeltaStream25fps(resnet=resnet, input_dim=resnet.backend_out, encoder_embed_dim=cfg.patch_embeding_dim, av_corr=cfg.av_corr)
        self.encoder = TransformerEncoder(cfg, skip_pos_conv=self.skip_pos_conv)

    def forward(self, video, video_lengths=None):
        features_video = self.delta_stream_25fps(video, None, video_lengths)
        features_video, _ = self.encoder(features_video, padding_mask=None, layer=None)
        return features_video


class TokenProj(nn.Module):
    """把 DeltaEncoder 特征投影到 LLM token 嵌入空间."""
    def __init__(self, d_model=1280, output_dim=2048):
        super(TokenProj, self).__init__()
        self.ln_post = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, output_dim)
        )

    def forward(self, x):
        output = self.proj(self.ln_post(x))
        return output


# =====================================================================
# DeltaEncoder: Visual Delta Encoder 主模块
# =====================================================================

class DeltaEncoder(nn.Module):
    """Visual Delta Encoder: encoder (DeltaEncoderBackbone) + token_proj_1x / token_proj_6x."""

    def __init__(self, cfg_file='HumanOmniSpeaker/delta_encoder.json', weights_path=None):
        super(DeltaEncoder, self).__init__()
        cfg = omegaconf.OmegaConf.load(cfg_file)

        self.encoder = DeltaEncoderBackbone(cfg)
        # 每帧 1 token / 每帧 6 token 两种输出密度的投影头
        self.token_proj_1x = TokenProj(d_model=1024, output_dim=cfg.feature_out_dim)
        self.token_proj_6x = TokenProj(d_model=1024, output_dim=cfg.feature_out_dim)

        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        current = self.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in current and current[k].shape != v.shape:
                print(f"[DeltaEncoder] skip shape-mismatched key: {k} "
                      f"(ckpt {tuple(v.shape)} vs model {tuple(current[k].shape)})")
                continue
            filtered[k] = v
        self.load_state_dict(filtered, strict=False)
