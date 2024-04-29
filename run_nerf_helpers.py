import math
from typing import Optional
from torch import Tensor
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# # TODO: remove this dependency
from torch import searchsorted

from matplotlib import pyplot as plt

# for mamba
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

from einops import rearrange, repeat
from timm.models.layers import DropPath

# Misc
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class CrossMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj_1 = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_2 = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, inputs, inference_params=None):
        """
        hidden_states: (B, L, D)
        input2: (B, L, D)
        Returns: same shape as hidden_states
        """
        hidden_states, input2 = inputs
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        x = rearrange(
            self.in_proj_1.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj_1.bias is not None:
            x = x + rearrange(self.in_proj_1.bias.to(dtype=x.dtype), "d -> d 1")

        z = rearrange(
            self.in_proj_2.weight @ rearrange(input2, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj_2.bias is not None:
            z = z + rearrange(self.in_proj_2.bias.to(dtype=z.dtype), "d -> d 1")
        xz = torch.cat([x, z], dim=1)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba_type == "v1":
                A_b = -torch.exp(self.A_b_log.float())
                out = bimamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    A_b,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )    
            elif self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
                if not self.if_devide_out:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                else:
                    out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight, self.out_proj.bias)

            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        if self.init_layer_scale is not None:
                out = out * self.gamma    
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class CrossBlock(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: tuple(input1, input2).
                input1: (B, N, C) <==> x
                input2: (B, N, C) <==> z
            residual: hidden_states = Mixer(LN(residual))
        """
        input1, input2 = hidden_states
        if not self.fused_add_norm:
            if residual is None:
                residual = input2
            else:
                residual = residual + self.drop_path(input2)
            
            input2 = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                input2, residual = fused_add_norm_fn(
                    input2,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                input2, residual = fused_add_norm_fn(
                    self.drop_path(input2),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer((input1, input2), inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
# Model
def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    mode='self',
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if mode == 'self':
        mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    elif mode == 'cross':
        mixer_cls = partial(CrossMamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
        block = CrossBlock(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    else:
        raise ValueError(f"Unknown mode {mode}")
    block.layer_idx = layer_idx
    return block

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)      # 透明度,一个值
            self.rgb_linear = nn.Linear(W//2, 3)     # rgb颜色，三个值
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        """
        Args:
            x: (bs * n_points, 90). 63 for points, 27 for viewdirs.
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # h: (bs * n_points, W)
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)        # alpha只与xyz有关
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)      # rgb与xyz和direct都有关
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)   #  (bs * n_points, W//2)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

class NeRF_RGB(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, alpha_model=None):
        """ 
        """
        super(NeRF_RGB, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            # self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.alpha_model = alpha_model

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            with torch.no_grad():
                alpha = self.alpha_model(x)[...,3][...,None]
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

class NeRF_mamba(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, use_viewdirs=False, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, residual_in_fp32=False, 
                 fused_add_norm=False, device=None, dtype=None):
        """ 
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NeRF_mamba, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.pts_embed = nn.Linear(input_ch, W)
        self.pts_linears = nn.ModuleList(
            [
                create_block(
                    d_model=W,
                    ssm_cfg={'expand': 1},
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(1)
            ])
        
        self.norm_alpha = (nn.LayerNorm if not rms_norm else RMSNorm)(
            W, eps=norm_epsilon, **factory_kwargs
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList(
            [
                create_block(
                    d_model=input_ch_views + W,
                    ssm_cfg={'expand': 1},
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(1)
            ])
        self.norm_rgb = (nn.LayerNorm if not rms_norm else RMSNorm)(
            input_ch_views + W, eps=norm_epsilon, **factory_kwargs
        )
        
        if use_viewdirs:
            # TODO: 处理不使用viewdirs的情况
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(input_ch_views + W, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        print(f'Congratulations! You are using NeRF + Mamba model!')

    def forward(self, x, inference_params=None):
        """
        args:
            x: (bs, n_points, c)
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        h = self.pts_embed(h)
        residual = None
        for i, layer in enumerate(self.pts_linears):
            h, residual = layer(h, residual, inference_params=inference_params)

        if not self.fused_add_norm:
            residual = (h + residual) if residual is not None else h
            h = self.norm_alpha(residual.to(dtype=self.norm_alpha.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_alpha, RMSNorm) else layer_norm_fn
            h = fused_add_norm_fn(
                h,
                self.norm_alpha.weight,
                self.norm_alpha.bias,
                eps=self.norm_alpha.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)

            h = torch.cat([h, input_views], -1)
            residual = None
        
            for i, layer in enumerate(self.views_linears):
                h, residual = layer(h, residual, inference_params=inference_params)

            if not self.fused_add_norm:
                residual = (h + residual) if residual is not None else h
                h = self.norm_rgb(residual.to(dtype=self.norm_rgb.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_alpha, RMSNorm) else layer_norm_fn
                h = fused_add_norm_fn(
                    h,
                    self.norm_rgb.weight,
                    self.norm_rgb.bias,
                    eps=self.norm_rgb.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self):
        print('Not Implemented Yet')
        return 

class NeRF_CA_mamba(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, use_viewdirs=False, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, residual_in_fp32=False, 
                 fused_add_norm=False, device=None, dtype=None):
        """ 
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NeRF_CA_mamba, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.pts_embed = nn.Linear(input_ch, W)
        self.view_embed = nn.Linear(input_ch_views, W)
        self.ca_fusion = nn.MultiheadAttention(embed_dim=W, num_heads=8, batch_first=True, **factory_kwargs)
        self.norm1 = nn.LayerNorm(W)
        self.dropout1 = nn.Dropout(0.1)

        self.mamba_block = create_block(
                    d_model=W,
                    ssm_cfg={'expand': 1},
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=0,
                    **factory_kwargs,
                )
        
        self.norm_alpha = (nn.LayerNorm if not rms_norm else RMSNorm)(
            W, eps=norm_epsilon, **factory_kwargs
        )
        
        if use_viewdirs:
            # TODO: 处理不使用viewdirs的情况
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        print(f'Congratulations! You are using Mamba **v3** model!')

    def forward(self, x, inference_params=None):
        """
        args:
            x: (bs, n_points, c)
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # Cross Attention
        q, kv = self.pts_embed(input_pts), self.view_embed(input_views)
        feature_fused = self.ca_fusion(q, kv, kv, need_weights=False)[0]
        feature_fused = self.dropout1(feature_fused) + q
        h = self.norm1(feature_fused)

        # Mamba with Norm
        residual = None
        h, residual = self.mamba_block(h, residual, inference_params=inference_params)
        if not self.fused_add_norm:
            residual = (h + residual) if residual is not None else h
            h = self.norm_alpha(residual.to(dtype=self.norm_alpha.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_alpha, RMSNorm) else layer_norm_fn
            h = fused_add_norm_fn(
                h,
                self.norm_alpha.weight,
                self.norm_alpha.bias,
                eps=self.norm_alpha.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self):
        print('Not Implemented Yet')
        return 

class NeRF_Vim(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, use_viewdirs=False, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, residual_in_fp32=False, 
                 fused_add_norm=False, device=None, dtype=None):
        """ 
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NeRF_Vim, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.pts_embed = nn.Linear(input_ch, W)
        # TODO: release this comment
        drop_path_rate = 0.1
        depth = 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.pts_linears = nn.ModuleList(
            [
                create_block(
                    d_model=W,
                    ssm_cfg={'expand': 1},
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=False,       # if True, use 'v1'
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                    **factory_kwargs,
                )
                for i in range(1)
            ])
        
        self.norm_alpha = (nn.LayerNorm if not rms_norm else RMSNorm)(
            W, eps=norm_epsilon, **factory_kwargs
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList(
            [
                create_block(
                    d_model=W + input_ch_views,
                    ssm_cfg={'expand': 1},
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=False,       # if True, use 'v1'
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                    **factory_kwargs,
                )
                for i in range(1)
            ])
        self.norm_rgb = (nn.LayerNorm if not rms_norm else RMSNorm)(
            input_ch_views + W, eps=norm_epsilon, **factory_kwargs
        )
        
        if use_viewdirs:
            # TODO: 处理不使用viewdirs的情况
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(input_ch_views + W, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        print(f'Congratulations! You are using **Mamba + ViM** model! bimamba_type = v2')

    def forward(self, x, inference_params=None):
        """
        args:
            x: (bs, n_points, c)
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        h = self.pts_embed(h)
        residual = None
        for i, layer in enumerate(self.pts_linears):
            h, residual = layer(h, residual, inference_params=inference_params)

        if not self.fused_add_norm:
            residual = (h + residual) if residual is not None else h
            h = self.norm_alpha(residual.to(dtype=self.norm_alpha.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_alpha, RMSNorm) else layer_norm_fn
            h = fused_add_norm_fn(
                h,
                self.norm_alpha.weight,
                self.norm_alpha.bias,
                eps=self.norm_alpha.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)

            h = torch.cat([h, input_views], -1)
            residual = None
        
            for i, layer in enumerate(self.views_linears):
                h, residual = layer(h, residual, inference_params=inference_params)

            if not self.fused_add_norm:
                residual = (h + residual) if residual is not None else h
                h = self.norm_rgb(residual.to(dtype=self.norm_rgb.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_alpha, RMSNorm) else layer_norm_fn
                h = fused_add_norm_fn(
                    h,
                    self.norm_rgb.weight,
                    self.norm_rgb.bias,
                    eps=self.norm_rgb.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self):
        print('Not Implemented Yet')
        return 

class NeRF_VimCm(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, use_viewdirs=False, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, residual_in_fp32=False, 
                 fused_add_norm=False, device=None, dtype=None):
        """ 
        NeRF + Vision Mamba + Visin Cross Mamba
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NeRF_VimCm, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.pts_embed = nn.Linear(input_ch, W)

        # TODO: release this comment
        drop_path_rate = 0.1
        depth = 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.pts_linears = nn.ModuleList(
            [
                create_block(
                    d_model=W,
                    ssm_cfg={'expand': 1},
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=False,       # if True, use 'v1'
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                    **factory_kwargs,
                )
                for i in range(1)
            ])
        
        self.norm_alpha = (nn.LayerNorm if not rms_norm else RMSNorm)(
            W, eps=norm_epsilon, **factory_kwargs
        )

        self.views_embed = nn.Linear(input_ch_views, W, **factory_kwargs)
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList(
            [
                create_block(
                    d_model=W,
                    ssm_cfg={'expand': 1},
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    mode='cross',
                    if_bimamba=False,       # if True, use 'v1'
                    bimamba_type='v2',
                    drop_path=inter_dpr[i],
                    if_devide_out=True,
                    init_layer_scale=None,
                    **factory_kwargs,
                )
                for i in range(1)
            ])
        self.norm_rgb = (nn.LayerNorm if not rms_norm else RMSNorm)(
            W, eps=norm_epsilon, **factory_kwargs
        )
        
        if use_viewdirs:
            # TODO: 处理不使用viewdirs的情况
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        print(f'Congratulations! You are using **NeRF_VimCm** model! bimamba type = v2')

    def forward(self, x, inference_params=None):
        """
        args:
            x: (bs, n_points, c)
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        h = self.pts_embed(h)
        residual = None
        for i, layer in enumerate(self.pts_linears):
            h, residual = layer(h, residual, inference_params=inference_params)

        if not self.fused_add_norm:
            residual = (h + residual) if residual is not None else h
            h = self.norm_alpha(residual.to(dtype=self.norm_alpha.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_alpha, RMSNorm) else layer_norm_fn
            h = fused_add_norm_fn(
                h,
                self.norm_alpha.weight,
                self.norm_alpha.bias,
                eps=self.norm_alpha.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)

            input_views = self.views_embed(input_views)
            residual = None
        
            for i, layer in enumerate(self.views_linears):
                h, residual = layer((h, input_views), residual, inference_params=inference_params)

            if not self.fused_add_norm:
                residual = (h + residual) if residual is not None else h
                h = self.norm_rgb(residual.to(dtype=self.norm_rgb.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_alpha, RMSNorm) else layer_norm_fn
                h = fused_add_norm_fn(
                    h,
                    self.norm_rgb.weight,
                    self.norm_rgb.bias,
                    eps=self.norm_rgb.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self):
        print('Not Implemented Yet')
        return 

class NeRF_CA_Vim(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, use_viewdirs=False, ssm_cfg=None, norm_epsilon=1e-5, rms_norm=False, residual_in_fp32=False, 
                 fused_add_norm=False, device=None, dtype=None):
        """ 
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super(NeRF_CA_Vim, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.pts_embed = nn.Linear(input_ch, W)
        self.view_embed = nn.Linear(input_ch_views, W)
        self.ca_fusion = nn.MultiheadAttention(embed_dim=W, num_heads=8, batch_first=True, **factory_kwargs)
        self.norm1 = nn.LayerNorm(W)
        self.dropout1 = nn.Dropout(0.1)
        # =======================For Vision Mamba ===========================
        # TODO: release this comment
        drop_path_rate = 0.1
        depth = 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.mamba_block = create_block(
                    d_model=W,
                    ssm_cfg={'expand': 1},
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=0,
                    if_bimamba=False,       # if True, use 'v1'
                    bimamba_type='v2',
                    drop_path=inter_dpr[0],
                    if_devide_out=True,
                    init_layer_scale=None,
                    **factory_kwargs,
                )
        
        self.norm_alpha = (nn.LayerNorm if not rms_norm else RMSNorm)(
            W, eps=norm_epsilon, **factory_kwargs
        )
        
        if use_viewdirs:
            # TODO: 处理不使用viewdirs的情况
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
        print(f'Congratulations! You are using Mamba **Cross Attention + ViM** model!')

    def forward(self, x, inference_params=None):
        """
        args:
            x: (bs, n_points, c)
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # Cross Attention
        q, kv = self.pts_embed(input_pts), self.view_embed(input_views)
        feature_fused = self.ca_fusion(q, kv, kv, need_weights=False)[0]
        feature_fused = self.dropout1(feature_fused) + q
        h = self.norm1(feature_fused)

        # Mamba with Norm
        residual = None
        h, residual = self.mamba_block(h, residual, inference_params=inference_params)
        if not self.fused_add_norm:
            residual = (h + residual) if residual is not None else h
            h = self.norm_alpha(residual.to(dtype=self.norm_alpha.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_alpha, RMSNorm) else layer_norm_fn
            h = fused_add_norm_fn(
                h,
                self.norm_alpha.weight,
                self.norm_alpha.bias,
                eps=self.norm_alpha.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self):
        print('Not Implemented Yet')
        return 

# Ray helpers
def get_rays(H, W, focal, c2w, device=torch.device('cpu')):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1).to(device)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # i: H x W, j: H x W
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1) # dirs: H x W x 3
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))  # H x W x 3
    return rays_o, rays_d


def get_rays_by_coord_np(H, W, focal, c2w, coords):
    i, j = (coords[:,0]-W*0.5)/focal, -(coords[:,1]-H*0.5)/focal
    dirs = np.stack([i,j,-np.ones_like(i)],-1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False, device=torch.device('cpu')):
    """
    Args:
        bins [torch.Tensor]: (bs, 63), bs=4096
        weights [torch.Tensor]: (bs, 62)
        N_samples [int]: 128
        det [bool]: False by default
        pytest [bool]: False by default
    """
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (bs, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])     # 也在0到1之间, (bs, N_samples)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        # 如果 pytest 为真，那么 u 将被固定的随机数覆盖，这对于测试是有用的，因为它可以确保每次运行的结果都是一样的。
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous().to(device)
    inds = searchsorted(cdf, u, right=True)         # 这个函数是找到u在离散cdf中对应的右区间点
    below = torch.max(torch.zeros_like(inds-1), inds-1)  # 找到左区间点，以0为最小值
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)  # 找到右区间点，以0为最小值
    inds_g = torch.stack([below, above], -1)  # (bs, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]    # (bs, N_samples, len(bins))
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)     # (bs, len(bins)) -> (bs, 1, len(bins)) -> (bs, N_samples, len(bins)) -> (bs, N_samples, 2)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)   # (bs, 63) -> (bs, 1, 63) -> (bs, N_samples, 63) -> (bs, N_samples, 2)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)     # 小于1e-5的设为1，否则为该区间对应的概率值
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, device=torch.device('cpu')):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
        noise = noise.to(device)

    # 计算公式3 [bs, 64]
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # 后面这部分就是Ti，前面是alpha，这个就是论文上的那个权重w [bs,64]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    rgb_var = torch.sum(weights[...,None] * (rgb - rgb_map.reshape(-1, 1, 3)) ** 2, -2)
    depth_var = torch.sum(weights * (z_vals - depth_map.reshape(-1, 1)) ** 2, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map


def sample_sigma(rays_o, rays_d, viewdirs, network, z_vals, network_query, device=torch.device('cpu')):
    # N_rays = rays_o.shape[0]
    # N_samples = len(z_vals)
    # z_vals = z_vals.expand([N_rays, N_samples])

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    raw = network_query(pts, viewdirs, network)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    sigma = F.relu(raw[...,3])

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, device=device)

    return rgb, sigma, depth_map


def visualize_sigma(sigma, z_vals, filename):
    plt.plot(z_vals, sigma)
    plt.xlabel('z_vals')
    plt.ylabel('sigma')
    plt.savefig(filename)
    return
