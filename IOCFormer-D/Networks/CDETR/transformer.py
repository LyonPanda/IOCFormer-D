# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .attention import MultiheadAttention
import numpy as np
import cv2
from pvt_v2 import PyramidVisionTransformerV2
from functools import partial
from timm.models.vision_transformer import _cfg
from mmseg.ops import resize
from AgentAttention import *

class Agent_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        if True:
            memory2=src.clone()
            memory2=memory2.permute(1,2,0).reshape(bs,c,h,w)
            memory2[memory2<0]=0
            memory2=memory2.mean(1).detach().cpu().numpy()
            # memory2=memory2[10].detach().cpu().numpy()
            for ii in range(bs):
                feat_map=memory2[ii]
                feat_map[feat_map<0]=0
                feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-5)
                feat_map = (feat_map * 255).astype(np.uint8)
                feat_map = cv2.applyColorMap(feat_map, cv2.COLORMAP_AUTUMN)
                cv2.imwrite("./visual_dm/visal_feat2/{}.png".format(ii), feat_map)
        hs, references = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                      pos=pos_embed, query_pos=query_embed)
        return hs, references


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class Transformer_featmerge(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, feat_pre=True, two_layers=False, with_norm=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder_featmerge(encoder_layer, num_encoder_layers, encoder_norm, d_model, feat_pre, two_layers, with_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model)

        self._reset_parameters()
        
        self.pvtv2_depth0 = PyramidVisionTransformerV2(img_size=256,
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
        
        self.pretrained = True
        
        if self.pretrained:
            self.load_model('./weights/pvt_v2_b0.pth') 
        
        #depth—merge
        self.layer_dep0 = nn.Conv2d(1, 3, kernel_size=1)
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(160, 64)
        self.linear4 = nn.Linear(256, 64)

        #agent_attention
        self.RGB_agent_layer_norm_1_1 = nn.LayerNorm(256)
        self.RGB_agent_layer_norm_1_2 = nn.LayerNorm(256)
        self.RGB_agent_layer_norm_2_1 = nn.LayerNorm(256)
        self.RGB_agent_layer_norm_2_2 = nn.LayerNorm(256)

        self.Depth_agent_layer_norm_1_1 = nn.LayerNorm(256)
        self.Depth_agent_layer_norm_1_2 = nn.LayerNorm(256)
        self.Depth_agent_layer_norm_2_1 = nn.LayerNorm(256)
        self.Depth_agent_layer_norm_2_2 = nn.LayerNorm(256)
        
        self.RGB_agent_mlp_1 = Agent_Mlp(in_features=256, hidden_features=256*4, act_layer=nn.GELU, drop=0.)
        self.RGB_agent_mlp_2 = Agent_Mlp(in_features=256, hidden_features=256*4, act_layer=nn.GELU, drop=0.)

        self.Depth_agent_mlp_1 = Agent_Mlp(in_features=256, hidden_features=256*4, act_layer=nn.GELU, drop=0.)
        self.Depth_agent_mlp_2 = Agent_Mlp(in_features=256, hidden_features=256*4, act_layer=nn.GELU, drop=0.)

        self.agent_attn_1 = AgentAttention(dim=256, num_patches=256, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0, sr_ratio=1, agent_num=49)
        self.agent_attn_2 = AgentAttention(dim=256, num_patches=256, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0, sr_ratio=1, agent_num=49)
        
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
    
    def load_model(self, pretrained):
        pretrained_dict = torch.load(pretrained)
        model_dict = self.pvtv2_depth0.state_dict()
        print("Loading pretrained parameters from {}".format(pretrained))
        for k, v in pretrained_dict.items():
            if k in model_dict:
                print(f"Loading parameter: {k}")
            else:
                print(f"Skipping parameter: {k}")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.pvtv2_depth0.load_state_dict(model_dict)
        print("Pretrained parameters loaded successfully.")

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, depth, mask, query_embed, pos_embed, density_feat):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, density_f=density_feat)
        
        #depth_feat0.shape:[bs,32,64,64], depth_feat1.shape:[bs,64,32,32], depth_feat2.shape:[bs,160,16,16], depth_feat3.shape:[bs,256,8,8]
        depth_feat0, depth_feat1, depth_feat2, depth_feat3 = self.pvtv2_depth0(self.layer_dep0(depth))
        
        depth_feat0 = self.linear1(depth_feat0.flatten(2).transpose(1, 2))
        depth_feat0 = depth_feat0.transpose(1, 2).reshape(bs, 64, 64, 64)
        
        depth_feat1 = self.linear2(depth_feat1.flatten(2).transpose(1, 2))
        depth_feat1 = depth_feat1.transpose(1, 2).reshape(bs, 64, 32, 32)
        depth_feat1 = resize(depth_feat1, size=(64, 64),mode='bilinear',align_corners=False)
        
        depth_feat2 = self.linear3(depth_feat2.flatten(2).transpose(1, 2))
        depth_feat2 = depth_feat2.transpose(1, 2).reshape(bs, 64, 16, 16)
        depth_feat2 = resize(depth_feat2, size=(64, 64),mode='bilinear',align_corners=False)
        
        depth_feat3 = self.linear4(depth_feat3.flatten(2).transpose(1, 2))
        depth_feat3 = depth_feat3.transpose(1, 2).reshape(bs, 64, 8, 8)
        depth_feat3 = resize(depth_feat3, size=(64, 64),mode='bilinear',align_corners=False)
        
        depth_featmerge = torch.cat([depth_feat0, depth_feat1, depth_feat2, depth_feat3], dim=1)
        depth_featmerge = resize(depth_featmerge, size=(16, 16),mode='bilinear',align_corners=False)
        depth_featmerge = depth_featmerge.flatten(2).permute(2, 0, 1) #depth_featmerge.shape:[256, bs, 256]
        
        #agent_attention
        memory = memory.permute(1,0,2).contiguous()
        depth_featmerge = depth_featmerge.permute(1,0,2).contiguous()

        #agent_attention_1
        RGB_feat_1 = self.RGB_agent_layer_norm_1_1(memory)
        Depth_feat_1 = self.Depth_agent_layer_norm_1_1(depth_featmerge)
        RGB_feat_1 , Depth_feat_1 = self.agent_attn_1(RGB_feat_1, Depth_feat_1, 16, 16)
        RGB_feat_1 = memory + RGB_feat_1
        Depth_feat_1 = depth_featmerge + Depth_feat_1
        RGB_feat_1 = RGB_feat_1 + self.RGB_agent_mlp_1(self.RGB_agent_layer_norm_1_2(RGB_feat_1))
        Depth_feat_1 = Depth_feat_1 + self.Depth_agent_mlp_1(self.Depth_agent_layer_norm_1_2(Depth_feat_1))
        
        #agent_attention_2
        RGB_feat_2 = self.RGB_agent_layer_norm_2_1(RGB_feat_1)
        Depth_feat_2 = self.Depth_agent_layer_norm_2_1(Depth_feat_1)
        RGB_feat_2 , Depth_feat_2 = self.agent_attn_2(RGB_feat_2, Depth_feat_2, 16, 16)
        RGB_feat_2 = RGB_feat_1 + RGB_feat_2
        Depth_feat_2 = Depth_feat_1 + Depth_feat_2
        RGB_feat_2 = RGB_feat_2 + self.RGB_agent_mlp_2(self.RGB_agent_layer_norm_2_2(RGB_feat_2))
        Depth_feat_2 = Depth_feat_2 + self.Depth_agent_mlp_2(self.Depth_agent_layer_norm_2_2(Depth_feat_2))

        RGB_final_feat = RGB_feat_2.permute(1,0,2).contiguous()
        Depth_final_feat = Depth_feat_2.permute(1,0,2).contiguous()
        fuse_feat = RGB_final_feat + Depth_final_feat

        # import pdb; pdb.set_trace()
        if False:
            memory2=src.clone()
            memory2=memory2.permute(1,2,0).reshape(bs,c,h,w)
            memory2[memory2<0]=0
            memory2=memory2.mean(1).detach().cpu().numpy()
            # memory2=memory2[10].detach().cpu().numpy()
            for ii in range(bs):
                feat_map=memory2[ii]
                feat_map[feat_map<0]=0
                feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min() + 1e-5)
                feat_map = (feat_map * 255).astype(np.uint8)
                feat_map = cv2.applyColorMap(feat_map, cv2.COLORMAP_JET)
                cv2.imwrite("./visual_dm/visal_feat/{}.png".format(ii), feat_map)
                
        hs, references = self.decoder(tgt, fuse_feat, memory_key_padding_mask=mask,
                                      pos=pos_embed, query_pos=query_embed)
        return hs, references


class TransformerEncoder_featmerge(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=512, feat_pre=True, two_layers=False, with_norm=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        if two_layers:
            self.layers_density=nn.ModuleList(
                [nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_model, d_model, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)) for i in range(num_layers)]
                )
        else:
            self.layers_density=nn.ModuleList(
                [nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)) for i in range(num_layers)]
                )
        self.with_norm=with_norm
        if with_norm:
            self.norm_density=nn.LayerNorm(d_model)
        self.feat_pre=feat_pre
        print(two_layers,with_norm)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                density_f: Optional[Tensor] = None):
        output = src

        assert density_f is not None
        # assert len(self.layers)==len(density_f)
        output_density=density_f
        for i, layer in enumerate(self.layers):
            if self.feat_pre: #False
                output_density=self.layers_density[i](output_density)
                # print(output.shape, output_density.shape)
                if self.with_norm:
                    output=output+self.norm_density(output_density.flatten(2).permute(2, 0, 1))
                else:
                    output=output+output_density.flatten(2).permute(2, 0, 1)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if not self.feat_pre: #True
                output_density=self.layers_density[i](output_density)
                if self.with_norm: #True
                    output=output+self.norm_density(output_density.flatten(2).permute(2, 0, 1))
                else:
                    output=output+output_density.flatten(2).permute(2, 0, 1)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)
        reference_points = reference_points_before_sigmoid.sigmoid().transpose(0, 1)

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2].transpose(0, 1)  # [num_queries, batch_size, 2]

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output) #pos_transformation.shape:torch.Size([700, bs, 256])

            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return [torch.stack(intermediate).transpose(1, 2), reference_points]

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before: #False
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed=None,
                     is_first=False):

        ## for debug
        # import pdb; pdb.set_trace()
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256 (d_model)
        q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape 

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first: #True
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):
        if self.normalize_before: #False
            raise NotImplementedError
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed,
                                 is_first)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    if args.transformer_flag=="merge":
        return Transformer_featmerge(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            num_queries=args.num_queries,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    elif args.transformer_flag=="merge2":
        return Transformer_featmerge(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            num_queries=args.num_queries,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            feat_pre=False,
        )
    elif args.transformer_flag=="merge3":
        return Transformer_featmerge(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            num_queries=args.num_queries,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
            feat_pre=False,
            two_layers=True,
            with_norm=True,
        )
    else:
        return Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            num_queries=args.num_queries,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")