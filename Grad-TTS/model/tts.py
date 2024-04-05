# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch
import numpy as np

# from model import monotonic_align
from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility

from matcha.models.components.flow_matching import CFM
from matcha.models.components.decoder import TimestepEmbedding

from vits.models import StochasticDurationPredictor

class GradTTS(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale, decoder_type="gradtts", matcha_config=[None, None], ifsdp=False, ifzeroshot=False, ifdecodercondition=False, ifuncondition=False, decoderconditiontype="ti", ifCFG=False):
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.decoder_type = decoder_type
        if decoder_type=="matchatts":
            ifRoPE = True
        else:
            ifRoPE = False
        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size, ifRoPE=ifRoPE)
        if decoder_type=="gradtts":
            self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)
        elif decoder_type=="matchatts":
            decoder, cfm = matcha_config
            self.decoder = CFM(
                in_channels=2 * n_feats,
                out_channel=n_feats,
                cfm_params=cfm,
                decoder_params=decoder,
                n_spks=n_spks,
                spk_emb_dim=spk_emb_dim,
            )
        self.ifsdp = ifsdp
        if self.ifsdp:
            self.sdp = StochasticDurationPredictor(n_feats, n_feats, 3, 0.5, 4, gin_channels=spk_emb_dim)
            
        self.ifzeroshot = ifzeroshot
        self.speaker_emb = None
        if n_spks > 1:
            if self.ifzeroshot:
                self.speaker_emb = torch.nn.Sequential(
                    torch.nn.Linear(256, spk_emb_dim), 
            )
        
        self.ifdecodercondition = ifdecodercondition
        self.decoderconditiontype = decoderconditiontype
        if self.ifdecodercondition:
            time_embed_dim = decoder["channels"][0]*4
            in_channels = 2*n_feats + (spk_emb_dim if n_spks > 1 else 0)
            outnch = time_embed_dim if self.decoderconditiontype=="ti" else in_channels
            self.ed_decoding_embedding = torch.nn.Sequential(
                torch.nn.Linear(12, outnch),
                torch.nn.Tanh(),
            )
                
            if self.decoderconditiontype=="ti":
                self.ed_time_mlp = TimestepEmbedding(
                    in_channels=in_channels,
                    time_embed_dim=time_embed_dim,
                    act_fn="silu",
                )
            
        self.ifuncondition = ifuncondition
        self.ifCFG = ifCFG
                
    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, ed, se, temperature=1.0, stoc=False, spk=None, length_scale=1.0, gscale=2.0, random_seed=None):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment
        
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            if self.ifzeroshot:
                spk_embedding = self.speaker_emb(se) # se: speaker embedding
            else:
                spk_embedding = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        if self.ifuncondition:
            mu_x, logw, x_mask = self.encoder(x, x_lengths, spk, ed=None)
        else:
            mu_x, logw, x_mask = self.encoder(x, x_lengths, spk, ed)
            if self.ifCFG:
                mu_x_unc, _, _ = self.encoder(x, x_lengths, spk, ed=None)
            
        if self.ifsdp:
            logw = self.sdp(mu_x, x_mask, g=spk_embedding.unsqueeze(-1), reverse=True, noise_scale=1.0)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        if self.ifCFG:
            mu_y_unc = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x_unc.transpose(1, 2))
            mu_y_unc = mu_y_unc.transpose(1, 2)
            cond2 = [mu_y_unc, gscale]
        else:
            cond2 = None
        encoder_outputs = mu_y[:, :, :y_max_length]

        if self.decoder_type=="gradtts":
            # Sample latent representation from terminal distribution N(mu_y, I)
            z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
            # Generate sample by performing reverse dynamics
            decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk_embedding)
        elif self.decoder_type=="matchatts":
            cond = None
            if self.ifdecodercondition:
                cond = self.ed_decoding_embedding(ed)
                if self.decoderconditiontype=="ti":
                    time = torch.arange(ed.shape[1]).to(ed.device)
                    time = self.decoder.estimator.time_embeddings(time).unsqueeze(0)
                    cond = cond + self.ed_time_mlp(time)
                    cond = (cond*x_mask.transpose(1,2)).sum(1)/x_mask.sum((2))
                elif self.decoderconditiontype=="fe":
                    aligned_cond = torch.matmul(attn.squeeze(1).transpose(1, 2), cond*x_mask.transpose(1,2))
                    cond = aligned_cond.transpose(1, 2)
                
            decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, spks=spk_embedding, cond=cond, cond2=cond2, random_seed=random_seed)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, duration, ed, se, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.
            
        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            if self.ifzeroshot:
                spk = self.speaker_emb(se) # se: speaker embedding
            else:
                spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        ed_condition = True
        if self.ifuncondition:
            ed_condition = False
        elif self.ifCFG: # Classifier Free Guidance
            if np.random.random()<0.1:
                ed_condition = False
        else:
            pass
        if ed_condition:
            mu_x, logw, x_mask = self.encoder(x, x_lengths, spk, ed)
        else:
            mu_x, logw, x_mask = self.encoder(x, x_lengths, spk, ed=None)
            
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)

        attn = torch.zeros((*duration.shape, y.shape[2]))
        for i in range(attn.shape[0]):
            start = 0
            for j in range(attn.shape[1]):
                now = duration[i][j]
                attn[i, j, start:start+now] = 1.0
                start += now
        attn = attn.to(mu_x.device)
        
        if self.ifsdp:
            w = attn.unsqueeze(1).detach().sum(3)
            l_length = self.sdp(mu_x, x_mask, w, g=spk.unsqueeze(-1))
            dur_loss = (l_length / torch.sum(x_mask)).sum()
        else:
            logw_ = torch.log(1e-8 + duration.unsqueeze(1)) * x_mask
            logw_ = logw_.to(mu_x.device)

            # Compute loss between predicted log-scaled durations and those obtained from MAS
            dur_loss = duration_loss(logw, logw_, x_lengths)

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        
        cond = None
        if self.ifdecodercondition:
            cond = self.ed_decoding_embedding(ed)
            if self.decoderconditiontype=="ti":
                time = torch.arange(ed.shape[1]).to(ed.device)
                time = self.decoder.estimator.time_embeddings(time).unsqueeze(0)
                cond = cond + self.ed_time_mlp(time)
                cond = (cond*x_mask.transpose(1,2)).sum(1)/x_mask.sum((2))
            elif self.decoderconditiontype=="fe":
                aligned_cond = torch.matmul(attn.squeeze(1).transpose(1, 2), cond*x_mask.transpose(1,2))
                cond = aligned_cond.transpose(1, 2)

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, spk, cond=cond)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return dur_loss, prior_loss, diff_loss
