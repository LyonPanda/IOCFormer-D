import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=49, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.RGB_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.RGB_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.RGB_attn_drop = nn.Dropout(attn_drop)
        self.RGB_proj = nn.Linear(dim, dim)
        self.RGB_proj_drop = nn.Dropout(proj_drop)
        
        self.Depth_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.Depth_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.Depth_attn_drop = nn.Dropout(attn_drop)
        self.Depth_proj = nn.Linear(dim, dim)
        self.Depth_proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.RGB_dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.Depth_dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        
        self.RGB_an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.RGB_na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.RGB_ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.RGB_aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.RGB_ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.RGB_wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.RGB_an_bias, std=.02)
        trunc_normal_(self.RGB_na_bias, std=.02)
        trunc_normal_(self.RGB_ah_bias, std=.02)
        trunc_normal_(self.RGB_aw_bias, std=.02)
        trunc_normal_(self.RGB_ha_bias, std=.02)
        trunc_normal_(self.RGB_wa_bias, std=.02)
        
        self.Depth_an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.Depth_na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.Depth_ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.Depth_aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.Depth_ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.Depth_wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.Depth_an_bias, std=.02)
        trunc_normal_(self.Depth_na_bias, std=.02)
        trunc_normal_(self.Depth_ah_bias, std=.02)
        trunc_normal_(self.Depth_aw_bias, std=.02)
        trunc_normal_(self.Depth_ha_bias, std=.02)
        trunc_normal_(self.Depth_wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.RGB_pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.Depth_pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, H, W):
        b, n, c = x.shape #b=bs, n=256, c=256
        num_heads = self.num_heads #num_heads=8
        head_dim = c // num_heads #head_dim=32
        
        #q.shape:[bs, 256, 256]
        RGB_q = self.RGB_q(x)
        Depth_q = self.Depth_q(y)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            #kv.shape:[2, bs, 256, 256]
            RGB_kv = self.RGB_kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3).contiguous()
            Depth_kv = self.Depth_kv(y).reshape(b, -1, 2, c).permute(2, 0, 1, 3).contiguous()
            
        RGB_k, RGB_v = RGB_kv[0], RGB_kv[1]
        Depth_k, Depth_v = Depth_kv[0], Depth_kv[1]
        
        #agent_tokens.shape:[bs, 49, 256] , 49 = agent_num
        RGB_agent_tokens = self.RGB_pool(RGB_q.reshape(b, H, W, c).permute(0, 3, 1, 2).contiguous()).reshape(b, c, -1).permute(0, 2, 1).contiguous()
        Depth_agent_tokens = self.Depth_pool(Depth_q.reshape(b, H, W, c).permute(0, 3, 1, 2).contiguous()).reshape(b, c, -1).permute(0, 2, 1).contiguous()
        
        #q,k,v = torch.Size([bs, 8, 256, 32])
        RGB_q = RGB_q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        RGB_k = RGB_k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        RGB_v = RGB_v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        Depth_q = Depth_q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        Depth_k = Depth_k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        Depth_v = Depth_v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
                
        #agent_tokens.shape:torch.Size([bs, 8, 49, 32])
        RGB_agent_tokens = RGB_agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        Depth_agent_tokens = Depth_agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio) #(16,16)
        
        #cross_attn1:Depth->RGB
        RGB_position_bias1 = nn.functional.interpolate(self.RGB_an_bias, size=kv_size, mode='bilinear')
        RGB_position_bias1 = RGB_position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        RGB_position_bias2 = (self.RGB_ah_bias + self.RGB_aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        RGB_position_bias = RGB_position_bias1 + RGB_position_bias2
        RGB_agent_attn = self.softmax((Depth_agent_tokens * self.scale) @ RGB_k.transpose(-2, -1) + RGB_position_bias)
        RGB_agent_attn = self.RGB_attn_drop(RGB_agent_attn)
        RGB_agent_v = RGB_agent_attn @ RGB_v

        RGB_agent_bias1 = nn.functional.interpolate(self.RGB_na_bias, size=self.window_size, mode='bilinear')
        RGB_agent_bias1 = RGB_agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        RGB_agent_bias2 = (self.RGB_ha_bias + self.RGB_wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        RGB_agent_bias = RGB_agent_bias1 + RGB_agent_bias2
        RGB_q_attn = self.softmax((RGB_q * self.scale) @ Depth_agent_tokens.transpose(-2, -1) + RGB_agent_bias)
        RGB_q_attn = self.RGB_attn_drop(RGB_q_attn)
        x = RGB_q_attn @ RGB_agent_v

        x = x.transpose(1, 2).contiguous().reshape(b, n, c)
        RGB_v = RGB_v.transpose(1, 2).contiguous().reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2).contiguous()
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.RGB_dwc(RGB_v).permute(0, 2, 3, 1).contiguous().reshape(b, n, c)
    
        #cross_attn2:RGB->Depth
        Depth_position_bias1 = nn.functional.interpolate(self.Depth_an_bias, size=kv_size, mode='bilinear')
        Depth_position_bias1 = Depth_position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        Depth_position_bias2 = (self.Depth_ah_bias + self.Depth_aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        Depth_position_bias = Depth_position_bias1 + Depth_position_bias2
        Depth_agent_attn = self.softmax((RGB_agent_tokens * self.scale) @ Depth_k.transpose(-2, -1) + Depth_position_bias)
        Depth_agent_attn = self.Depth_attn_drop(Depth_agent_attn)
        Depth_agent_v = Depth_agent_attn @ Depth_v

        Depth_agent_bias1 = nn.functional.interpolate(self.Depth_na_bias, size=self.window_size, mode='bilinear')
        Depth_agent_bias1 = Depth_agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        Depth_agent_bias2 = (self.Depth_ha_bias + self.Depth_wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        Depth_agent_bias = Depth_agent_bias1 + Depth_agent_bias2
        Depth_q_attn = self.softmax((Depth_q * self.scale) @ RGB_agent_tokens.transpose(-2, -1) + Depth_agent_bias)
        Depth_q_attn = self.Depth_attn_drop(Depth_q_attn)
        y = Depth_q_attn @ Depth_agent_v

        y = y.transpose(1, 2).contiguous().reshape(b, n, c)
        Depth_v = Depth_v.transpose(1, 2).contiguous().reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2).contiguous()
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        y = y + self.Depth_dwc(Depth_v).permute(0, 2, 3, 1).contiguous().reshape(b, n, c)
        
        x = self.RGB_proj(x)
        x = self.RGB_proj_drop(x)
        y = self.Depth_proj(y)
        y = self.Depth_proj_drop(y)
        return x, y 