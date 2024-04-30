from torchsummary import summary
from thop import profile
import torch
from run_nerf_helpers import *

model_type = None  # 'VimCm'

netdepth = 8
netwidth = 256
output_ch = 5
use_viewdirs = True
device = 'cuda'
skips = [4]

input_ch = 63
input_ch_views = 27

if model_type == 'v1':
    model = NeRF_mamba(D=netdepth, W=netwidth,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
# TODO: add model type v2
elif model_type == 'v3':
    model = NeRF_CA_mamba(D=netdepth, W=netwidth,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
elif model_type == 'Vim':
    model = NeRF_Vim(D=netdepth, W=netwidth,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
elif model_type == 'VimCm':
    model = NeRF_VimCm(D=netdepth, W=netwidth,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
elif model_type == 'CA_Vim':
    model = NeRF_CA_Vim(D=netdepth, W=netwidth,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
else:
    print(f'you are using original NeRF now')
    model = NeRF(D=netdepth, W=netwidth,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs).to(device)
    
netdepth_fine = 8
netwidth_fine = 256

if model_type == 'v1':
    model_fine = NeRF_mamba(D=netdepth_fine, W=netwidth_fine,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
elif model_type == 'v3':
    model_fine = NeRF_CA_mamba(D=netdepth_fine, W=netwidth_fine,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
elif model_type == 'Vim':
    model_fine = NeRF_Vim(D=netdepth_fine, W=netwidth_fine,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
elif model_type == 'VimCm':
    model_fine = NeRF_VimCm(D=netdepth, W=netwidth,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
elif model_type == 'CA_Vim':
    model_fine = NeRF_CA_Vim(D=netdepth, W=netwidth,
                input_ch=input_ch, output_ch=output_ch,
                input_ch_views=input_ch_views, use_viewdirs=use_viewdirs,
                device=device,
                ).to(device)
else:
    print(f'you are using original NeRF now')
    model_fine = NeRF(D=netdepth_fine, W=netwidth_fine,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=use_viewdirs).to(device)
    
input_size = (64, 90)  # 例如，对于224x224的RGB图像
input_fine_size = (128, 90)

# 获取参数数量
summary(model, (64, 90))

# 获取FLOPs
input = torch.randn(*input_size).cuda()
macs, params = profile(model, inputs=(input, ))
print('FLOPs: %.2f' % (macs * 2))  # 乘以2是因为乘法和加法都算作一次FLOP