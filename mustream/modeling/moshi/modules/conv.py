# Portions of this implementation are adapted from Moshi
# (https://github.com/kyutai-labs/moshi), licensed under the MIT License.

"""Causal streaming 1-D convolution (StreamingConv1d), used by the streaming
MERT feature extractor (modeling/streaming/streaming_mert.py)."""

from dataclasses import dataclass
import itertools
import typing as tp
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from .streaming import StreamingModule, State


CONV_NORMALIZATIONS = frozenset(["none", "weight_norm"])
M = tp.TypeVar('M', bound=nn.Module)


def apply_parametrization_norm(module: M, norm: str = "none") -> M:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(
            nn.Conv1d(*args, **kwargs), norm
        )
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        return x


@dataclass
class _StreamingConv1dState(State):
    previous: torch.Tensor
    first: torch.Tensor

    def reset(self, reset_mask: torch.Tensor):
        super().reset(reset_mask)
        self.previous[:] = torch.where(reset_mask.view(-1, 1, 1), torch.zeros_like(self.previous), self.previous)
        self.first[:] = torch.where(reset_mask, torch.ones_like(self.first), self.first)


class StreamingConv1d(StreamingModule[_StreamingConv1dState]):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = "constant",
    ):
        super().__init__()
        assert pad_mode in ['constant', 'replicate'], pad_mode
        self.pad_mode = pad_mode
        self.causal = causal
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamingConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )

    @property
    def _stride(self) -> int:
        return self.conv.conv.stride[0]

    @property
    def _kernel_size(self) -> int:
        return self.conv.conv.kernel_size[0]

    @property
    def _effective_kernel_size(self) -> int:
        dilation = self.conv.conv.dilation[0]
        return (
            self._kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations

    @property
    def _padding_total(self) -> int:
        return self._effective_kernel_size - self._stride

    def _init_streaming_state(self, batch_size: int) -> _StreamingConv1dState:
        assert self.causal, "non-causal StreamingConv1d does not support streaming inference"
        stride = self._stride
        # Effective kernel size accounting for dilation.
        kernel = self._effective_kernel_size
        param = next(iter(self.parameters()))
        dtype = param.dtype
        device = param.device
        previous = torch.zeros(batch_size, self.conv.conv.in_channels, kernel - stride,
                               dtype=dtype, device=device)
        first = torch.ones(batch_size, device=device, dtype=torch.bool)
        return _StreamingConv1dState(batch_size, device, previous, first)

    def forward(self, x):
        B, C, T = x.shape
        S = self._stride
        assert T > 0 and T % S == 0, "Steps must be multiple of stride"
        if not self.causal:
            # Symmetric, non-causal padding: no streaming state involved, the model
            # needs the full sequence (including future context) up front.
            padding_total = self._padding_total
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            return self.conv(F.pad(x, (padding_left, padding_right)))
        state = self._streaming_state
        if state is None:
            state = self._init_streaming_state(B)
        TP = state.previous.shape[-1]
        if TP and self.pad_mode == 'replicate':
            assert T >= TP, "Not enough content to pad streaming."
            init = x[..., :1]
            state.previous[:] = torch.where(
                state.first.view(-1, 1, 1) & state.exec_mask.view(-1, 1, 1),
                init,
                state.previous)
        if TP:
            x = torch.cat([state.previous, x], dim=-1)
        y = self.conv(x)
        if TP:
            state.previous[:] = torch.where(
                state.exec_mask.view(-1, 1, 1),
                x[..., -TP:],
                state.previous)
            if self.pad_mode == 'replicate':
                state.first = torch.where(
                    state.exec_mask,
                    torch.zeros_like(state.first),
                    state.first,
                )
        return y


def test():
    """Chunked streaming StreamingConv1d must reproduce a single full pass."""
    torch.manual_seed(1234)
    device = "cpu"
    if torch.cuda.is_available():
        # Avoid the cuda optimizations that would take place on single precision
        # floats for convolutions.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda:0"

    kernel_sizes = [1, 3, 4, 8, 15, 16]
    strides = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    chin = 6
    chout = 12

    for kernel, stride in itertools.product(kernel_sizes, strides):
        if stride > kernel:
            continue
        conv = StreamingConv1d(chin, chout, kernel, stride, causal=True).to(device)

        for frames in [1, 4, 8, 32, 54, 65, 128]:
            print(f"ksize {kernel} strides {stride} frames {frames}")
            batch_size = 3
            length = frames * stride
            x = torch.randn(batch_size, chin, length).to(device)
            y = conv(x)
            for chunk_frames in [1, 2, 8]:
                if frames % chunk_frames != 0:
                    continue
                ys = []
                chunk_length = chunk_frames * stride
                with conv.streaming(batch_size):
                    for offset in range(0, length, chunk_length):
                        ys.append(conv(x[..., offset : offset + chunk_length]))
                y_stream = torch.cat(ys, dim=-1)
                y = y[..., : y_stream.shape[-1]]
                assert y.shape == y_stream.shape, (y.shape, y_stream.shape)
                delta = (y_stream - y).norm() / y.norm()
                assert delta <= 1e-6, delta
                assert frames == y_stream.shape[-1], (frames, y_stream.shape)


if __name__ == "__main__":
    with torch.no_grad():
        test()
