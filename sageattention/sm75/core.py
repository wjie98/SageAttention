import warnings
from typing import Any, Optional

import torch

from .. import _fused_sm75 as _fused
from . import sm75_compile
from .quant import per_warp_int8, per_warp_int8_varlen, sub_mean


def _sdpa_dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str, is_causal: bool, sm_scale: Optional[float]):
    if tensor_layout == "HND":
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=sm_scale)
    if tensor_layout == "NHD":
        o = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=is_causal,
            scale=sm_scale,
        )
        return o.transpose(1, 2).contiguous()
    raise ValueError(f"Unsupported tensor_layout: {tensor_layout}")


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    seq_dim = 1 if tensor_layout == "NHD" else 2
    if not return_lse and q.size(seq_dim) <= 1024 and k.size(seq_dim) <= 1024:
        return _sdpa_dense(q, k, v, tensor_layout, is_causal, sm_scale)

    return sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran="per_warp",
        sm_scale=sm_scale,
        return_lse=return_lse,
        pv_accum_dtype="fp32",
    )


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)
    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    if smooth_k:
        km = k.mean(dim=0, keepdim=True)
        k = k - km

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    if max_seqlen_q >= 512:
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_varlen(
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            BLKQ=64,
            WARPQ=16,
            BLKK=64,
        )
        o = torch.empty_like(q)
        sm75_compile.qk_int8_sv_f16_varlen_accum_f32_attn(
            q_int8,
            k_int8,
            v.to(torch.float16).contiguous(),
            o,
            q_scale,
            k_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            int(is_causal),
            sm_scale,
        )
        return o[..., :head_dim_og]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    _fused.varlen_attention_fwd_cuda(q, k, v, cu_seqlens_q, cu_seqlens_k, o, max_seqlen_q, sm_scale, int(is_causal))
    return o[..., :head_dim_og]


def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_warp",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran == "per_warp", "sm75 backend supports qk_quant_gran='per_warp'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    tensor_layout_id = 0 if tensor_layout == "NHD" else 1
    is_causal_id = 1 if is_causal else 0
    qk_quant_gran_id = 2
    return_lse_id = 1 if return_lse else 0

    head_dim_og = q.size(-1)
    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if tensor_layout_id == 0 else 2
    nh_dim = 2 if tensor_layout_id == 0 else 1

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        nqheads = q.size(nh_dim)
        nkheads = k.size(nh_dim)
        q_per_kv_heads = nqheads // nkheads
        if q_per_kv_heads > 1:
            km_broadcast = torch.repeat_interleave(km, q_per_kv_heads, dim=nh_dim)
        else:
            km_broadcast = km
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_broadcast.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_broadcast.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    q_int8, q_scale, k_int8, k_scale = per_warp_int8(
        q,
        k,
        km,
        tensor_layout=tensor_layout,
        BLKQ=64,
        WARPQ=16,
        BLKK=64,
        fuse_qk=(km is None and (is_causal or (tensor_layout == "HND" and q.size(-1) == 64))),
    )

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_dtype in ["fp32", "fp16+fp32"] and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == "fp32":
        v = v.to(torch.float16)
        lse = sm75_compile.qk_int8_sv_f16_accum_f32_attn(
            q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout_id, is_causal_id, qk_quant_gran_id, sm_scale, return_lse_id
        )
    elif pv_accum_dtype == "fp16":
        if smooth_v:
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
            lse = sm75_compile.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(
                q_int8, k_int8, smoothed_v, o, q_scale, k_scale, vm, tensor_layout_id, is_causal_id, qk_quant_gran_id, sm_scale, return_lse_id
            )
        else:
            v = v.to(torch.float16)
            lse = sm75_compile.qk_int8_sv_f16_accum_f16_attn(
                q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout_id, is_causal_id, qk_quant_gran_id, sm_scale, return_lse_id
            )
    elif pv_accum_dtype == "fp16+fp32":
        v = v.to(torch.float16)
        lse = sm75_compile.qk_int8_sv_f16_accum_f16_attn_inst_buf(
            q_int8, k_int8, v, o, q_scale, k_scale, tensor_layout_id, is_causal_id, qk_quant_gran_id, sm_scale, return_lse_id
        )
    else:
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    return o
