# SPDX-License-Identifier: Apache-2.0
"""The Flash-Next checkpoint's image path (`hibana/model/flashnext`): the Qwen3-VL vision tower behind the Qwen4Exp
multimodal wrapper — a 27-block ViT of width 1152 (16 heads, head_dim 72) over 16-pixel patches, 2 x 2 patch blocks
merged per token, and the `merger` projector into the language model's 2560-wide embedding space.

`preprocess` smart-resizes each image to 32-pixel-aligned sides between 65,536 and 16,777,216 pixels (the checkpoint's
`preprocessor_config.json`: factor = patch 16 x merge 2, bicubic antialiased resize, mean/std 0.5), then layes out its
patches in raster order. `QwenVision.encode` folds the reference's Conv3d patch embedding (temporal kernel 2, fed the
frame twice for images) into one Linear, adds the learned 48 x 48 position table resampled to the patch grid
(bilinear, corners aligned), runs the blocks with per-frame full attention under axial 2-D rotary (theta 10,000, h and
w frequencies interleaved over the whole head as HF lays them out), and merges each 2 x 2 patch block with
LayerNorm(1152) -> fc1 4608 -> erf-GELU -> fc2 2560. The reference (`transformers` 5.15.1 `qwen3_vl`, which
`Qwen4ExpVisionModel` upstream copies and vLLM's flash-next model inherits) instead feeds the tower patches in
spatial-merge-block order and merges with a plain reshape; full per-frame attention is permutation-equivariant, so the
raster layout with coordinates matching each patch gives the same embeddings (proven to < 1e-2 by
`tools/vision_parity.py`). An image of (h x w) patches becomes h * w / 4 embeddings, which replace the
`<|image_pad|>` placeholders the API inserts between `<|vision_start|>` and `<|vision_end|>`. Weights are the
checkpoint's BF16 `model.visual.*` tensors (about 0.9 GiB); the arithmetic follows the reference modules in BF16 with
FP32 norms and rotary products.
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
        proj = get(f"{m}.patch_embed.proj.weight")
        self.patch = proj.sum(dim=2).reshape(WIDTH, 3 * PATCH * PATCH)  # sums the two temporal slices
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
        inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, self.head_dim // 2, 2, dtype=torch.float32,
                                                      device=device) / (self.head_dim // 2)))  # 18 frequencies
        self.inv_freq = inv_freq
        if self.patch.shape != (WIDTH, 3 * PATCH * PATCH) or self.pos.shape != (POS_GRID, POS_GRID, WIDTH) \
                or self.m_fc1_w.shape != (MERGER_HIDDEN, MERGER_HIDDEN) \
                or self.m_fc2_w.shape != (OUT_WIDTH, MERGER_HIDDEN):
            raise ValueError("unexpected vision tower shapes")
        self.hidden = OUT_WIDTH

    @property
    def bytes(self) -> int:
        tensors = [self.patch, self.patch_bias, self.pos, self.m_norm, self.m_norm_b, self.m_fc1_w, self.m_fc1_b,
                   self.m_fc2_w, self.m_fc2_b, *(t for b in self.blocks for t in b)]
        return sum(t.numel() * t.element_size() for t in tensors)

    def _pos_embeds(self, rows: int, cols: int) -> torch.Tensor:
        """The learned 48 x 48 table resampled (bilinear, corners aligned) to a rows x cols patch grid,
        [rows * cols, WIDTH] FP32. The reference gathers four taps per patch and sums them; one bilinear kernel writes
        the same values (align_corners matches its scale)."""
        pos = self.pos.permute(2, 0, 1).unsqueeze(0)  # [1, 1152, 48, 48]
        interp = F.interpolate(pos, size=(rows, cols), mode="bilinear", align_corners=True)
        return interp[0].flatten(1).t()

    def _cos_sin(self, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Axial rotary cos/sin for a rows x cols raster grid, [rows * cols, head_dim] FP32 each: 18 h-frequencies then
        18 w-frequencies, the whole head rotated (the reference concatenates the same halves twice)."""
        h = torch.arange(rows, device=self.device, dtype=torch.float32).repeat_interleave(cols)  # row per patch
        w = torch.arange(cols, device=self.device, dtype=torch.float32).repeat(rows)  # column per patch
        freqs = torch.stack([h, w], dim=-1)[..., None] * self.inv_freq  # [N, 2, 18]
        freqs = freqs.flatten(1)  # [N, 36]
        emb = torch.cat([freqs, freqs], dim=-1)  # [N, 72]: h-freqs, w-freqs, repeated over the head's halves
        return emb.cos(), emb.sin()

    @torch.inference_mode()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """An image [3, H, W] (`preprocess`, H and W multiples of 32) -> [H/16 * W/16 / 4, 2560] BF16 embeddings."""
        x = pixels.to(self.device, torch.bfloat16)
        rows, cols = x.shape[1] // PATCH, x.shape[2] // PATCH
        patches = x.reshape(3, rows, PATCH, cols, PATCH).permute(1, 3, 0, 2, 4).reshape(rows * cols, -1)
        h = F.linear(patches, self.patch, self.patch_bias) + self._pos_embeds(rows, cols).to(torch.bfloat16)
        cos, sin = self._cos_sin(rows, cols)
        t, d = h.shape[0], self.head_dim
        for n1w, n1b, qkv_w, qkv_b, proj_w, proj_b, n2w, n2b, fc1_w, fc1_b, fc2_w, fc2_b in self.blocks:
            qkv = F.linear(F.layer_norm(h, (WIDTH,), n1w, n1b, LN_EPS), qkv_w, qkv_b).view(t, 3, HEADS, d).permute(1, 2, 0, 3)
            q, k = qkv[0:1], qkv[1:2]
            qf, kf = q.float(), k.float()  # rotary products in FP32, as the reference keeps them
            cos_h, sin_h = cos.unsqueeze(1), sin.unsqueeze(1)  # [t, 1, d]
            half = d // 2
            rot = lambda v: torch.cat((-v[..., half:], v[..., :half]), dim=-1)
            q = (qf * cos_h + rot(qf) * sin_h).to(torch.bfloat16)
            k = (kf * cos_h + rot(kf) * sin_h).to(torch.bfloat16)
            a = F.scaled_dot_product_attention(q, k, qkv[2:3])  # [1, heads, t, d]; full attention over the image
            h = h + F.linear(a[0].transpose(0, 1).reshape(t, WIDTH), proj_w, proj_b)
            h = h + F.linear(F.gelu(F.linear(F.layer_norm(h, (WIDTH,), n2w, n2b, LN_EPS), fc1_w, fc1_b),
                                    approximate="tanh"), fc2_w, fc2_b)
        return self._merge(h, rows, cols)

    @torch.inference_mode()
    def _merge(self, h: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """Tower features of a rows x cols raster grid -> the language model's embeddings, one per 2 x 2 patches."""
        n = F.layer_norm(h, (WIDTH,), self.m_norm, self.m_norm_b, LN_EPS)
        n = n.view(rows // MERGE, MERGE, cols // MERGE, MERGE, WIDTH).permute(0, 2, 1, 3, 4)
        n = n.reshape(-1, MERGER_HIDDEN)
        n = F.gelu(F.linear(n, self.m_fc1_w, self.m_fc1_b))  # erf GELU, as the reference merger
        return F.linear(n, self.m_fc2_w, self.m_fc2_b)
