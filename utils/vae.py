import torch
from torch import nn
from config.ExperimentCoMnfig import ExperimentConfig

from networks.NetworksImgCelebA import EncoderImg, DecoderImg
from networks.NetworksTextCelebA import EncoderText, DecoderText
from networks.ConvNetworksPolyMNIST import Encoder, Decoder
from networks.NetworksscMNC import scEncoder, scDecoder
from networks.ConvNetworksPolyMNIST import ResnetEncoder, ResnetDecoder
from networks.JointPrior import OrthogMat
# from networks.NetworksRatsspike import Encoder as RatsEncoder
# from networks.NetworksRatsspike import Decoder as RatsDecoder


def get_networks(cfg: ExperimentConfig) -> list[nn.ModuleList]:
    if cfg.dataset.name.startswith("PM"):
        if not cfg.model.use_resnets:
            encoders = nn.ModuleList(
                [
                    Encoder(cfg.model.latent_dim).to(cfg.model.device)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
            decoders = nn.ModuleList(
                [
                    Decoder(cfg.model.latent_dim).to(cfg.model.device)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
        else:
            encoders = nn.ModuleList(
                [
                    ResnetEncoder(cfg).to(cfg.model.device)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
            decoders = nn.ModuleList(
                [
                    ResnetDecoder(cfg).to(cfg.model.device)
                    for _ in range(cfg.dataset.num_views)
                ]
            )
    elif cfg.dataset.name.startswith("celeba"):
        encoders = nn.ModuleList(
            [
                EncoderImg(cfg).to(cfg.model.device),
                EncoderText(cfg).to(cfg.model.device),
            ]
        )
        decoders = nn.ModuleList(
            [
                DecoderImg(cfg).to(cfg.model.device),
                DecoderText(cfg).to(cfg.model.device),
            ]
        )
    elif cfg.dataset.name.startswith("sc"):
        original_dims = [1302, 39]
        encoders = nn.ModuleList(
            [
                scEncoder(original_dims[m], cfg.model.latent_dim, cfg.model.hidden_dim).to(cfg.model.device)
                for m in range(cfg.dataset.num_views)
            ]
        )
        decoders = nn.ModuleList(
            [
                scDecoder(original_dims[m], cfg.model.latent_dim, cfg.model.hidden_dim).to(cfg.model.device)
                for m in range(cfg.dataset.num_views)
            ]
        )
    else:
        raise NotImplementedError(
            "Unknown dataset/networks to create encoders and decoders for specified config"
        )
    
    if cfg.model.name == "jointprior":
      cov_mat = nn.ModuleList(
          [
            OrthogMat(cfg).to(cfg.model.device) for _ in range(cfg.dataset.num_views - 1)
          ]
        )
    else:
      cov_mat = torch.eye(cfg.model.latent_dim * cfg.dataset.num_views).to(cfg.model.device) 
      
    covariance = torch.eye(cfg.model.latent_dim * cfg.dataset.num_views).to(cfg.model.device) 
    mu = torch.zeros(cfg.model.latent_dim * cfg.dataset.num_views).to(cfg.model.device)
      
    return [encoders, decoders, cov_mat, covariance, mu]
