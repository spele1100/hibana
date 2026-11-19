# SPDX-License-Identifier: Apache-2.0
"""The Flash-Next checkpoint's image path (`hibana/model/flashnext`): the Qwen3-VL vision tower behind the Qwen4Exp
multimodal wrapper — a 27-block ViT of width 1152 (16 heads, head_dim 72) over 16-pixel patches with full per-frame
attention (no windows), 2 x 2 patch blocks merged per token, and the `merger` projector into the language model's
2560-wide embedding space.

`preprocess` smart-resizes each image to 32-pixel-aligned sides between 65,536 and 16,777,216 pixels (the
checkpoint's `preprocessor_config.json`: factor = patch 16 x merge 2, bicubic antialiased resize, mean/std 0.5).
`QwenVision.encode` runs the reference's (`transformers` 5.15.1 `qwen3_vl`, which upstream `Qwen4ExpVisionModel`
copies and vLLM's flash-next model inherits) exact arithmetic on its exact bits: patches in spatial-merge-block
order, the Conv3d patch embedding over the frame duplicated along its temporal axis, the learned 48 x 48 position
table resampled per patch (bilinear, corners aligned, four taps summed in FP32 with a true tensor division — a
scalar divisor would multiply by the FP32 reciprocal and drift an ulp), axial 2-D rotary (theta 10,000) with BF16
frequencies and angles exactly as the module's BF16 cast keeps them, sdpa full attention per frame, and the merger's
LayerNorm(1152) -> fc1 4608 -> erf GELU -> fc2 2560. Proven bit-identical to the reference on the RadixArk
checkpoint by `tools/vision_parity.py`. An image of (h x w) patches becomes h * w / 4 embeddings, which replace the
`<|image_pad|>` placeholders the API inserts between `<|vision_start|>` and `<|vision_end|>`. Weights are the
checkpoint's BF16 `model.visual.*` tensors (about 0.9 GiB); FP32 norms, rotary products and position sums, as there.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

TOWER = "model.visual"
PATCH = 16
TEMPORAL = 2  # the Conv3d patch kernel's depth; an image's frame is its two input slices
MERGE = 2  # spatial merge: 2 x 2 patches a token
WIDTH, HEADS, LAYERS, INTERMEDIATE = 1152, 16, 27, 4304
POS_GRID = 48  # the learned position table is num_position_embeddings 2304 = 48 x 48 patches
OUT_WIDTH = 2560
ROPE_THETA = 10_000.0  # the reference's default (Qwen3VLVisionRotaryEmbedding); the checkpoint config sets none
LN_EPS = 1e-6
MERGER_HIDDEN = WIDTH * MERGE**2
MIN_PIXELS, MAX_PIXELS = 65_536, 16_777_216  # preprocessor_config.json size shortest/longest edge
FACTOR = PATCH * MERGE  # smart resize aligns sides to 32 pixels
NORM_MEAN = NORM_STD = (0.5, 0.5, 0.5)
VISION_BYTES = 1_100_000_000  # the tower and projector (BF16) as loaded: the load's headroom
# One encode's device memory beyond its output (worst case a 16,384-patch image: qkv + rope temporaries). With room
# for the allocator's blocks
ENCODE_BYTES = 800 * 2**20

# tokenizer_config.json of the checkpoint: <|image_pad|> (config.json image_token_id), <|vision_start|>, <|vision_end|>
IMAGE_TOKEN, IMAGE_START, IMAGE_END = 248_056, 248_053, 248_054
# Part of every image's digest (api/media.py), so a cached or stored prefix is never resumed for pixels prepared
# another way; change it with `preprocess`.
IDENTITY = b"flashnext qwen3-vl tower: smart-resized to 32px-aligned sides of 65536-16777216 pixels, bicubic antialiased, 0.5/0.5 normalized, 16px patches raster, 2x2 merged\n"


def target_size(width: int, height: int, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """(w, h) pixels for a `width` x `height` image: the reference processor's `smart_resize` — sides rounded to
    32 pixels, aspect kept, area clamped to [min_pixels, max_pixels] (round, then floor when too large / ceil when
    too small, matching the reference's expressions). ValueError for absurd aspect ratios."""
    if max(width, height) / min(width, height) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(width, height) / min(width, height)}")
    h = round(height / FACTOR) * FACTOR
    w = round(width / FACTOR) * FACTOR
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / FACTOR) * FACTOR
        w = math.ceil(width * beta / FACTOR) * FACTOR
    return w, h


def image_tokens(width: int, height: int) -> int:
    """The embeddings `encode` makes of a `width` x `height` image (`target_size`; ValueError as there)."""
    w, h = target_size(width, height)
    return (h // PATCH) * (w // PATCH) // MERGE**2


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """An RGB image [H, W, 3] uint8 as the tower's input: resized (bicubic, antialiased) to its smart-resized size and
    normalized (mean/std 0.5), [3, h, w] BF16 on the CPU, patches in raster order."""
    w, h = target_size(rgb.shape[1], rgb.shape[0])
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    if t.shape[-2:] != (h, w):
        t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    mean, std = torch.tensor(NORM_MEAN).view(1, 3, 1, 1), torch.tensor(NORM_STD).view(1, 3, 1, 1)
    return ((t / 255.0 - mean) / std)[0].to(torch.bfloat16)


class QwenVision:
    """The tower and projector on the device (engine/contracts.VisionEncoder); `encode` turns one preprocessed image
    into embeddings."""

    image_token, image_start, image_end, identity = IMAGE_TOKEN, IMAGE_START, IMAGE_END, IDENTITY
    workspace_bytes = ENCODE_BYTES
    preprocess = staticmethod(preprocess)
    tokens = staticmethod(image_tokens)

    def __init__(self, w, device: torch.device):
        def get(name: str) -> torch.Tensor:
            return w.get(name).to(device)

        m = TOWER
        self.device = device
        # The reference's Conv3d [1152, 3, 2, 16, 16] reads each image's frame twice (temporal_patch_size 2); folded
        # into one [1152, 3 * 16 * 16] Linear over the frame. Raster patch layout matches the folded input.
        self.patch = get(f"{m}.patch_embed.proj.weight")  # [1152, 3, 2, 16, 16] conv3d, kept as the reference has it
        self.patch_bias = get(f"{m}.patch_embed.proj.bias")
        self.pos = get(f"{m}.pos_embed.weight").reshape(POS_GRID, POS_GRID, WIDTH).float()  # [48, 48, 1152]
        names = ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
                 "norm2.weight", "norm2.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias", "mlp.linear_fc2.weight",
                 "mlp.linear_fc2.bias")
        self.blocks = [[get(f"{m}.blocks.{i}.{n}") for n in names] for i in range(LAYERS)]
        self.m_norm, self.m_norm_b = get(f"{m}.merger.norm.weight"), get(f"{m}.merger.norm.bias")
        self.m_fc1_w, self.m_fc1_b = get(f"{m}.merger.linear_fc1.weight"), get(f"{m}.merger.linear_fc1.bias")
        self.m_fc2_w, self.m_fc2_b = get(f"{m}.merger.linear_fc2.weight"), get(f"{m}.merger.linear_fc2.bias")
        self.head_dim = WIDTH // HEADS
        # BF16 frequencies and angles: the reference's `to(dtype)` casts its module buffer, so cos/sin are BF16
        # values; the rotary products themselves run in FP32 (BF16 angles at large grids shift cos by ~0.1)
        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, self.head_dim // 2, 2, dtype=torch.float32,
                                                      device=device) / (self.head_dim // 2)))  # 18 frequencies
        self.inv_freq = inv_freq.to(torch.bfloat16)
        if self.patch.shape != (WIDTH, 3, TEMPORAL, PATCH, PATCH) or self.pos.shape != (POS_GRID, POS_GRID, WIDTH) \
                or self.m_fc1_w.shape != (MERGER_HIDDEN, MERGER_HIDDEN) \
                or self.m_fc2_w.shape != (OUT_WIDTH, MERGER_HIDDEN):
            raise ValueError("unexpected vision tower shapes")
        self.hidden = OUT_WIDTH

    @property
    def bytes(self) -> int:
        tensors = [self.patch, self.patch_bias, self.pos, self.m_norm, self.m_norm_b, self.m_fc1_w, self.m_fc1_b,
                   self.m_fc2_w, self.m_fc2_b, *(t for b in self.blocks for t in b)]
        return sum(t.numel() * t.element_size() for t in tensors)

    def _pos_embeds(self, r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """The learned 48 x 48 table resampled (bilinear, corners aligned) at each patch's (row, col) — the
        reference's own arithmetic: four taps a patch, BF16 table rows times FP32 weights, summed in FP32.
        `r` and `c` are the patches' coordinates as the sequence orders them."""
        side = POS_GRID
        def taps(idx: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
            src = (idx.float() * (side - 1)) / torch.tensor(max(size - 1, 1), device=idx.device)  # true division,
            # as the reference (a tensor divisor): `x / int` multiplies by the FP32 reciprocal and rounds twice
            floor = src.floor().long()
            return torch.stack([floor, (floor + 1).clamp(max=side - 1)], dim=1), torch.stack(
                [1 - (src - floor), (src - floor).clamp(max=1.0)], dim=1)
        ht, hw = taps(r, int(r.max()) + 1)
        wt, ww = taps(c, int(c.max()) + 1)
        idx = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
        w = (hw[:, :, None] * ww[:, None, :]).reshape(-1, 4)
        return (self.pos.flatten(0, 1)[idx] * w[:, :, None]).sum(1)  # the reference's own gather-and-sum

    def _cos_sin(self, r: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Axial rotary cos/sin for each patch's (row, column), [N, head_dim] BF16 each: 18 h-frequencies then 18
        w-frequencies, the whole head rotated (the reference concatenates the same halves twice). The int64 x BF16
        product keeps the reference's BF16 angles."""
        pid = torch.stack([r, c], dim=-1)  # [N, 2]
        freqs = (pid[..., None] * self.inv_freq).flatten(1)  # [N, 36]
        emb = torch.cat([freqs, freqs], dim=-1)  # [N, 72]
        return emb.cos(), emb.sin()

    @torch.inference_mode()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """An image [3, H, W] (`preprocess`, H and W multiples of 32) -> [H/16 * W/16 / 4, 2560] BF16 embeddings.
        Patches run in the reference's spatial-merge-block order ((block row, block column, row, column) within the
        grid), so every kernel sees the reference's own bits and the embeddings match it bitwise."""
        x = pixels.to(self.device, torch.bfloat16)
        rows, cols = x.shape[1] // PATCH, x.shape[2] // PATCH
        # (C, h/2, 2, 16, w/2, 2, 16) -> (h/2, w/2, 2, 2, C, 16, 16): the processor's own patch layout
        patches = x.reshape(3, rows // MERGE, MERGE, PATCH, cols // MERGE, MERGE, PATCH) \
            .permute(1, 4, 2, 5, 0, 3, 6).reshape(-1, 3, PATCH, PATCH)
        patches = patches[:, :, None].expand(-1, 3, TEMPORAL, PATCH, PATCH).contiguous()  # frame in both T slices
        h = F.conv3d(patches, self.patch, self.patch_bias).reshape(-1, WIDTH) \
            + self._pos_embeds(*self._coords(rows, cols)).to(torch.bfloat16)
        cos, sin = self._cos_sin(*self._coords(rows, cols))
        t, d = h.shape[0], self.head_dim
        for n1w, n1b, qkv_w, qkv_b, proj_w, proj_b, n2w, n2b, fc1_w, fc1_b, fc2_w, fc2_b in self.blocks:
            qkv = F.linear(F.layer_norm(h, (WIDTH,), n1w, n1b, LN_EPS), qkv_w, qkv_b).view(t, 3, HEADS, d)
            q, k, v = (t_.transpose(0, 1).unsqueeze(0) for t_ in qkv.permute(1, 0, 2, 3).unbind(0))  # [1, heads, t, d]
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            a = F.scaled_dot_product_attention(q, k, v)  # full attention over the image; one frame = one segment
            h = h + F.linear(a[0].transpose(0, 1).reshape(t, WIDTH), proj_w, proj_b)
            h = h + F.linear(F.gelu(F.linear(F.layer_norm(h, (WIDTH,), n2w, n2b, LN_EPS), fc1_w, fc1_b),
                                    approximate="tanh"), fc2_w, fc2_b)
        return self._merge(h)

    def _coords(self, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Each patch's (row, column) in spatial-merge-block sequence order."""
        r = torch.arange(rows, device=self.device).view(-1, 1, MERGE, 1).expand(-1, cols // MERGE, MERGE, MERGE)
        c = torch.arange(cols, device=self.device).view(1, -1, 1, MERGE).expand(rows // MERGE, -1, MERGE, MERGE)
        return r.flatten(), c.flatten()

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """[1, heads, t, d] under the axial rotary tables [t, d]: FP32 products, BF16 result, as the reference
        rotates (half-turned pairs over the whole head)."""
        cos, sin = cos[None, None].float(), sin[None, None].float()
        f = x.float()
        half = f.shape[-1] // 2
        rot = torch.cat((-f[..., half:], f[..., :half]), dim=-1)
        return (f * cos + rot * sin).to(x.dtype)

    @torch.inference_mode()
    def _merge(self, h: torch.Tensor) -> torch.Tensor:
        """Tower features in spatial-merge-block order -> the language model's embeddings, one per 2 x 2 patches
        (consecutive there): LayerNorm(1152) a patch, then fc1 4608 -> erf GELU -> fc2 2560."""
        n = F.layer_norm(h, (WIDTH,), self.m_norm, self.m_norm_b, LN_EPS).view(-1, MERGER_HIDDEN)
        n = F.gelu(F.linear(n, self.m_fc1_w, self.m_fc1_b))  # erf GELU, as the reference merger
        return F.linear(n, self.m_fc2_w, self.m_fc2_b)
