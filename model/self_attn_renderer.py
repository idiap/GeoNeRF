# GeoNeRF is a generalizable NeRF model that renders novel views
# without requiring per-scene optimization. This software is the 
# implementation of the paper "GeoNeRF: Generalizing NeRF with 
# Geometry Priors" by Mohammad Mahdi Johari, Yann Lepoittevin,
# and Francois Fleuret.

# Copyright (c) 2022 ams International AG

# This file is part of GeoNeRF.
# GeoNeRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# GeoNeRF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with GeoNeRF. If not, see <http://www.gnu.org/licenses/>.

# This file incorporates work covered by the following copyright and  
# permission notice:

    # Copyright 2020 Google LLC
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

def weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1.0 / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(stdv, stdv)


def masked_softmax(x, mask, **kwargs):
    x_masked = x.masked_fill(mask == 0, -float("inf"))

    return torch.softmax(x_masked, **kwargs)


## Auto-encoder network
class ConvAutoEncoder(nn.Module):
    def __init__(self, num_ch, S):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_ch, num_ch * 2, 3, stride=1, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch * 4, 3, stride=1, padding=1),
            nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(num_ch * 4, num_ch * 4, 3, stride=1, padding=1),
            nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(2),
        )

        # Decoder
        self.t_conv1 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch * 4, 4, stride=2, padding=1),
            nn.LayerNorm(S // 4, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv2 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 8, num_ch * 2, 4, stride=2, padding=1),
            nn.LayerNorm(S // 2, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        self.t_conv3 = nn.Sequential(
            nn.ConvTranspose1d(num_ch * 4, num_ch, 4, stride=2, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )
        # Output
        self.conv_out = nn.Sequential(
            nn.Conv1d(num_ch * 2, num_ch, 3, stride=1, padding=1),
            nn.LayerNorm(S, elementwise_affine=False),
            nn.ELU(alpha=1.0, inplace=True),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        conv1_out = x
        x = self.conv2(x)
        conv2_out = x
        x = self.conv3(x)

        x = self.t_conv1(x)
        x = self.t_conv2(torch.cat([x, conv2_out], dim=1))
        x = self.t_conv3(torch.cat([x, conv1_out], dim=1))

        x = self.conv_out(torch.cat([x, input], dim=1))

        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = masked_softmax(attn, mask, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x += residual

        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.transpose(1, 2).unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Renderer(nn.Module):
    def __init__(self, nb_samples_per_ray):
        super(Renderer, self).__init__()

        self.dim = 32
        self.attn_token_gen = nn.Linear(24 + 1 + 8, self.dim)

        ## Self-Attention Settings
        d_inner = self.dim
        n_head = 4
        d_k = self.dim // n_head
        d_v = self.dim // n_head
        num_layers = 4
        self.attn_layers = nn.ModuleList(
            [
                EncoderLayer(self.dim, d_inner, n_head, d_k, d_v)
                for i in range(num_layers)
            ]
        )

        ## Processing the mean and variance of input features
        self.var_mean_fc1 = nn.Linear(16, self.dim)
        self.var_mean_fc2 = nn.Linear(self.dim, self.dim)

        ## Setting mask of var_mean always enabled
        self.var_mean_mask = torch.tensor([1]).cuda()
        self.var_mean_mask.requires_grad = False

        ## For aggregating data along ray samples
        self.auto_enc = ConvAutoEncoder(self.dim, nb_samples_per_ray)

        self.sigma_fc1 = nn.Linear(self.dim, self.dim)
        self.sigma_fc2 = nn.Linear(self.dim, self.dim // 2)
        self.sigma_fc3 = nn.Linear(self.dim // 2, 1)

        self.rgb_fc1 = nn.Linear(self.dim + 9, self.dim)
        self.rgb_fc2 = nn.Linear(self.dim, self.dim // 2)
        self.rgb_fc3 = nn.Linear(self.dim // 2, 1)

        ## Initialization
        self.sigma_fc3.apply(weights_init)

    def forward(self, viewdirs, feat, occ_masks):
        ## Viewing samples regardless of batch or ray
        N, S, V = feat.shape[:3]
        feat = feat.view(-1, *feat.shape[2:])
        v_feat = feat[..., :24]
        s_feat = feat[..., 24 : 24 + 8]
        colors = feat[..., 24 + 8 : -1]
        vis_mask = feat[..., -1:].detach()

        occ_masks = occ_masks.view(-1, *occ_masks.shape[2:])
        viewdirs = viewdirs.view(-1, *viewdirs.shape[2:])

        ## Mean and variance of 2D features provide view-independent tokens
        var_mean = torch.var_mean(s_feat, dim=1, unbiased=False, keepdim=True)
        var_mean = torch.cat(var_mean, dim=-1)
        var_mean = F.elu(self.var_mean_fc1(var_mean))
        var_mean = F.elu(self.var_mean_fc2(var_mean))

        ## Converting the input features to tokens (view-dependent) before self-attention
        tokens = F.elu(
            self.attn_token_gen(torch.cat([v_feat, vis_mask, s_feat], dim=-1))
        )
        tokens = torch.cat([tokens, var_mean], dim=1)

        ## Adding a new channel to mask for var_mean
        vis_mask = torch.cat(
            [vis_mask, self.var_mean_mask.view(1, 1, 1).expand(N * S, -1, -1)], dim=1
        )
        ## If a point is not visible by any source view, force its masks to enabled
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)

        ## Taking occ_masks into account, but remembering if there were any visibility before that
        mask_cloned = vis_mask.clone()
        vis_mask[:, :-1] *= occ_masks
        vis_mask = vis_mask.masked_fill(vis_mask.sum(dim=1, keepdims=True) == 1, 1)
        masks = vis_mask * mask_cloned

        ## Performing self-attention
        for layer in self.attn_layers:
            tokens, _ = layer(tokens, masks)

        ## Predicting sigma with an Auto-Encoder and MLP
        sigma_tokens = tokens[:, -1:]
        sigma_tokens = sigma_tokens.view(N, S, self.dim).transpose(1, 2)
        sigma_tokens = self.auto_enc(sigma_tokens)
        sigma_tokens = sigma_tokens.transpose(1, 2).reshape(N * S, 1, self.dim)

        sigma_tokens = F.elu(self.sigma_fc1(sigma_tokens))
        sigma_tokens = F.elu(self.sigma_fc2(sigma_tokens))
        sigma = torch.relu(self.sigma_fc3(sigma_tokens[:, 0]))

        ## Concatenating positional encodings and predicting RGB weights
        rgb_tokens = torch.cat([tokens[:, :-1], viewdirs], dim=-1)
        rgb_tokens = F.elu(self.rgb_fc1(rgb_tokens))
        rgb_tokens = F.elu(self.rgb_fc2(rgb_tokens))
        rgb_w = self.rgb_fc3(rgb_tokens)
        rgb_w = masked_softmax(rgb_w, masks[:, :-1], dim=1)

        rgb = (colors * rgb_w).sum(1)

        outputs = torch.cat([rgb, sigma], -1)
        outputs = outputs.reshape(N, S, -1)

        return outputs
