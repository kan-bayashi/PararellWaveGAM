# -*- coding: utf-8 -*-

"""HiFi-GAN Modules.

This code is based on https://github.com/jik876/hifi-gan.

"""
from argparse import Namespace
import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F


from parallel_wavegan.layers import HiFiGANResidualBlock as ResidualBlock
from parallel_wavegan.utils import read_hdf5


def base_s3prl_setup(args):
    args.upstream_feature_selection = getattr(args, "upstream_feature_selection", None)
    args.upstream_model_config = getattr(args, "upstream_model_config", None)
    args.upstream_refresh = getattr(args, "upstream_refresh", False)
    args.upstream_ckpt = getattr(args, "upstream_ckpt", None)
    args.init_ckpt = getattr(args, "init_ckpt", None)
    args.verbose = getattr(args, "verbose", False)
    args.tile_factor = getattr(args, "tile_factor", 1)
    return args


class HuBERTREPRHiFiGANGenerator(torch.nn.Module):
    """HiFiGAN generator with HuBERT representation module."""

    def __init__(
        self,
        in_channels=1024,
        out_channels=1,
        channels=512,
        num_spk_embs=128,
        spk_emb_dim=128,
        spk_emb_inventory=None,
        concat_spk_emb=False,
        kernel_size=7,
        upsample_scales=(10, 8, 2, 2),
        upsample_kernel_sizes=(20, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        ckpt_path=None,
        layer_idx=9,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernal sizes for upsampling layers.
            resblock_kernal_sizes (list): List of kernal sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        self.num_spk_embs = num_spk_embs

        if self.num_spk_embs > 0:
            # self.spk_emb = torch.nn.Embedding(
            #     num_embeddings=num_spk_embs, embedding_dim=spk_emb_dim
            # )
            if spk_emb_inventory is None:
                self.spk_emb = torch.nn.Embedding(
                    num_embeddings=num_spk_embs, embedding_dim=spk_emb_dim
                )
            else:
                spk_emb = torch.load(spk_emb_inventory)
                self.spk_emb = torch.nn.Embedding.from_pretrained(spk_emb)
                self.spk_emb.requires_grad = False

            self.concat_spk_emb = concat_spk_emb
            if not concat_spk_emb:
                assert in_channels == spk_emb_dim
            else:
                in_channels = in_channels + spk_emb_dim

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernal size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                torch.nn.Sequential(
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                    torch.nn.ConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_kernel_sizes[i],
                        upsample_scales[i],
                        padding=upsample_scales[i] // 2 + upsample_scales[i] % 2,
                        output_padding=upsample_scales[i] % 2,
                    ),
                )
            ]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [
                    ResidualBlock(
                        kernel_size=resblock_kernel_sizes[j],
                        channels=channels // (2 ** (i + 1)),
                        dilations=resblock_dilations[j],
                        bias=bias,
                        use_additional_convs=use_additional_convs,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                    )
                ]
        self.output_conv = torch.nn.Sequential(
            # NOTE(kan-bayashi): follow official implementation but why
            #   using different slope parameter here? (0.1 vs. 0.01)
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                channels // (2 ** (i + 1)),
                out_channels,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # HuBERT
        assert ckpt_path is not None
        self.layer = layer_idx
        import fairseq
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.upstream = model[0].eval() # .cuda()
        self.task = task
        for k, p in self.upstream.named_parameters():
            logging.info(f"Setting {k}.requires_grad = False")
            p.requires_grad = False
        self.upstream_pretrained_params = copy.deepcopy(self.upstream.state_dict())

        # reset parameters
        self.reset_parameters()
        self.upstream.load_state_dict(self.upstream_pretrained_params)

    def forward(self, c, l):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input audio tensor (B, T, D).
            l (Tensor): Input text Tensor (B, 2, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """

        # convert idx to embedding
        if self.num_spk_embs > 0:
            assert l.size(1) == 2
            _, g_idx = l.long().split(1, dim=1)
            g = self.spk_emb(g_idx[:, 0, 0])

            # integrate global embedding
            if not self.concat_spk_emb:
                c = c + g.unsqueeze(2)
            else:
                g = g.unsqueeze(1).expand(-1, c.size(1), -1)  # (B, T, D)
                c = torch.cat([c, g], dim=-1)  # (B, T, D1 + D2)
                c = c.transpose(1, 2)  # (B, D', T)

        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0  # initialize
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)

        return c

    def extract_features(self, c):
        """Extract features from audio.

        Args:
            c (Tensor): Input audio tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, L, D).

        """

        f = c.squeeze(1)
        with torch.no_grad():
            if self.task.cfg.normalize:
                f = torch.nn.functional.layer_norm(f, f.shape)

            f, _ = self.upstream.extract_features(
                source=f,
                padding_mask=None,
                mask=False,
                output_layer=self.layer,
            )

        return f

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        if stats.endswith(".h5"):
            mean = read_hdf5(stats, "mean").reshape(-1)
            scale = read_hdf5(stats, "scale").reshape(-1)
        else:
            mean = np.load(stats)[0].reshape(-1)
            scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")

    def inference(self, c, l, normalize_before=False):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, D).
            l (Tensor or Int): Input spkid Tensor (1) or int.
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).

        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if not isinstance(l, torch.Tensor):
            l = torch.tensor(l, dtype=torch.long).to(c.device)

        l = l[None, :].repeat(2, 1)

        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.forward(c.unsqueeze(0), l.unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)
