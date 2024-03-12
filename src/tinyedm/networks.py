import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = np.sqrt(n.numel() / x.numel(), dtype=np.float32)
    return x / torch.add(eps, n, alpha=alpha)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(w)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight) / np.sqrt(fan_in, dtype=np.float32)
        x = F.conv2d(x, w, padding="same")
        return x

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
        )


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        w = torch.randn(out_features, in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(w)

    def forward(self, x: Tensor):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight) / np.sqrt(fan_in, dtype=np.float32)
        x = F.linear(x, w)
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}"


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest-exact")


def pixel_norm(x: Tensor, eps: float = 1e-4, dim=1) -> Tensor:
    return x / (torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + eps))


def mp_silu(x: Tensor) -> Tensor:
    return F.silu(x) / 0.596


def mp_add(a: Tensor, b: Tensor, t: float = 0.3) -> Tensor:
    scale = np.sqrt(t ** 2 + (1 - t) ** 2, dtype=np.float32)
    return ((1 - t) * a + t * b) / scale


class UncertaintyNet(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.linear1 = Linear(in_features + 1, hidden_features)
        self.linear2 = Linear(hidden_features, 1)

    def forward(self, x: Tensor):
        ones_tensor = torch.ones_like(x[:, 0:1])
        x = torch.cat((x, ones_tensor), dim=1)
        x = mp_silu(self.linear1(x))
        x = self.linear2(x)
        return x


class Scalelong(nn.Module):
    def __init__(self, dim, r=16):
        super(Scalelong, self).__init__()
        self.layer1 = Conv2d(dim + 1, int(dim // r), 1)
        self.layer2 = Conv2d(int(dim // r), dim, 1)

    def forward(self, inp):
        ones_tensor = torch.ones_like(inp[:, 0:1, :, :])
        inp = torch.cat((inp, ones_tensor), dim=1)
        gain = F.sigmoid(
            self.layer2(mp_silu(self.layer1(torch.mean(inp, dim=[2, 3], keepdim=True))))
        )
        return gain


class ClassEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.linear = Linear(num_embeddings, embedding_dim)

    def forward(self, class_labels: Tensor):
        class_emb = F.one_hot(class_labels.flatten(), self.num_embeddings)
        return self.linear(class_emb * np.sqrt(self.num_embeddings, dtype=np.float32))


class FourierEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.register_buffer("freqs", torch.randn(embedding_dim))
        self.register_buffer("phases", torch.rand(embedding_dim))

    def forward(self, x):
        x = torch.outer(x.flatten(), self.freqs) + self.phases
        x = torch.cos(2 * torch.pi * x) * np.sqrt(2, dtype=np.float32)
        return x


class Embedding(nn.Module):
    def __init__(
        self,
        fourier_dim: int,
        embedding_dim: int,
        num_classes: int | None = None,
        add_factor: float = 0.5,
    ):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.add_factor = add_factor
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.fourier_embed = FourierEmbedding(fourier_dim)
        self.sigma_embed = Linear(fourier_dim, embedding_dim)
        self.class_embed = None
        if num_classes is not None and num_classes != -1:
            self.class_embed = ClassEmbedding(num_classes, embedding_dim)

    def forward(self, sigmas, class_labels=None):
        with torch.cuda.amp.autocast(enabled=False):
            c_noise = sigmas.log() / 4
            fourier_embedding = self.fourier_embed(c_noise)
            embedding = self.sigma_embed(fourier_embedding)

            if class_labels is not None:
                if self.class_embed is None:
                    raise ValueError(
                        "class_labels is not None, but num_classes is None. "
                    )
                class_embedding = self.class_embed(class_labels)
                embedding = mp_add(embedding, class_embedding, self.add_factor)

            out = mp_silu(embedding)
        return fourier_embedding, out


class CosineAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        self.qkv_conv = Conv2d(embedding_dim, 3 * embedding_dim, 1)
        self.out_conv = Conv2d(embedding_dim, embedding_dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(x)  # (b, c, h, w) -> (b, 3*c, h, w)
        q, k, v = qkv.chunk(3, 1)  # (b, c, h*w)

        q = q.view(b, self.num_heads, self.head_dim, -1).transpose(2, 3)
        k = k.view(b, self.num_heads, self.head_dim, -1).transpose(2, 3)
        v = v.view(b, self.num_heads, self.head_dim, -1).transpose(2, 3)
        q, k, v = pixel_norm(q, dim=-1), pixel_norm(k, dim=-1), pixel_norm(v, dim=-1)

        y = F.scaled_dot_product_attention(q, k, v)  # (b, num_heads, h*w, head_dim)

        y = y.transpose(2, 3).reshape(b, -1, h, w)
        y = self.out_conv(y)

        out = mp_add(x, y)
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        down: bool,
        attention: bool,
        num_heads: int = 4,
        dropout_rate: float = 0.0,
        add_factor: float = 0.3,
    ):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.add_factor = add_factor

        self.resample = nn.AvgPool2d(kernel_size=2, stride=2) if down else nn.Identity()

        self.conv_1x1 = (
            Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.conv_3x3_1 = Conv2d(out_channels, out_channels, 3)
        self.conv_3x3_2 = Conv2d(out_channels, out_channels, 3)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.attention = (
            CosineAttention(out_channels, num_heads) if attention else nn.Identity()
        )

        # embedding layer
        self.embed = Linear(embedding_dim, out_channels)
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, input: Tensor, embedding: Tensor) -> Tensor:
        x = self.resample(input)
        x = self.conv_1x1(x)
        x = pixel_norm(x)

        # Residual branch
        res = x
        res = mp_silu(res)
        res = self.conv_3x3_1(res)

        with torch.cuda.amp.autocast(enabled=False):
            res = res * (self.embed(embedding) * self.gain + 1).unsqueeze(-1).unsqueeze(
                -1
            )
        res = mp_silu(res)
        res = self.dropout(res)
        res = self.conv_3x3_2(res)

        out = mp_add(x, res, self.add_factor)
        out = self.attention(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        up: bool,
        attention: bool,
        num_heads: int = 4,
        skip_channels: int = 0,
        dropout_rate: float = 0.0,
        add_factor: float = 0.3,
    ):
        super().__init__()

        self.add_factor = add_factor
        self.cat_factor = Scalelong(skip_channels) if skip_channels > 0 else None

        self.resample = Upsample() if up else nn.Identity()

        total_input_channels = in_channels + skip_channels
        self.conv_1x1 = (
            Conv2d(total_input_channels, out_channels, 1)
            if total_input_channels != out_channels
            else nn.Identity()
        )

        self.conv_3x3_1 = Conv2d(total_input_channels, out_channels, 3)
        self.conv_3x3_2 = Conv2d(out_channels, out_channels, 3)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = (
            CosineAttention(out_channels, num_heads) if attention else nn.Identity()
        )

        # embedding layer
        self.embed = Linear(embedding_dim, out_channels)
        self.gain = nn.Parameter(torch.ones(1))

    def forward(
        self, input: Tensor, embedding: Tensor, skip: Tensor | None = None
    ) -> Tensor:
        if skip is not None:
            input = torch.cat((input, skip * self.cat_factor(skip)), dim=1)
        x = self.resample(input)
        res = x
        x = self.conv_1x1(x)

        res = mp_silu(res)
        res = self.conv_3x3_1(res)

        with torch.cuda.amp.autocast(enabled=False):
            res = res * (self.embed(embedding) * self.gain + 1).unsqueeze(-1).unsqueeze(
                -1
            )
        res = mp_silu(res)
        res = self.dropout(res)
        res = self.conv_3x3_2(res)

        out = mp_add(x, res, self.add_factor)
        out = self.attention(out)
        return out


def get_encoder_blocks_types() -> tuple[str]:
    return (
        "Enc",
        "Enc",
        "Enc",
        "EncD",
        "Enc",
        "Enc",
        "Enc",
        "EncD",
        "EncA",
        "EncA",
        "EncA",
        "EncD",
        "EncA",
        "EncA",
        "EncA",
    )


def get_decoder_blocks_types() -> tuple[str]:
    return (
        "DecA",
        "Dec",
        "DecA",
        "DecA",
        "DecA",
        "DecA",
        "DecU",
        "DecA",
        "DecA",
        "DecA",
        "DecA",
        "DecU",
        "Dec",
        "Dec",
        "Dec",
        "Dec",
        "DecU",
        "Dec",
        "Dec",
        "Dec",
        "Dec",
    )


def get_encoder_out_channels() -> tuple[int]:
    return (192, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 576, 768, 768, 768)


def get_decoder_out_channels() -> tuple[int]:
    return (
        768,
        768,
        768,
        768,
        768,
        768,
        576,
        576,
        576,
        576,
        576,
        384,
        384,
        384,
        384,
        384,
        384,
        192,
        192,
        192,
        192,
    )


def get_skip_connections() -> tuple[int]:
    """The indices of decoder blocks that have skip connections."""
    return (
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    )


def get_skip_channels(
    encoder_out_channels: tuple[int],
    decoder_out_channels: tuple[int],
    skip_connections: tuple[bool],
) -> tuple[int]:
    skip_channels = np.zeros(len(decoder_out_channels), dtype=int)
    skip_channels[list(skip_connections)] = list(encoder_out_channels[::-1]) + [
        encoder_out_channels[0]
    ]  # input block for skip connections
    return tuple(skip_channels)


def build_encoder_blocks(block_types, out_channels, **kwargs):
    encoder_blocks = nn.ModuleList()
    in_channel = out_channels[0]
    for block_type, out_channel in zip(block_types, out_channels):
        down = block_type.endswith("D")
        attention = block_type.endswith("A")
        encoder_blocks.append(
            EncoderBlock(
                in_channels=in_channel,
                out_channels=out_channel,
                down=down,
                attention=attention,
                **kwargs,
            )
        )
        in_channel = out_channel

    return encoder_blocks


def build_decoder_blocks(block_types, out_channels, skip_channels, **kwargs):
    decoder_blocks = nn.ModuleList()
    in_channel = out_channels[0]
    for block_type, out_channel, skip_channel in zip(
        block_types, out_channels, skip_channels
    ):
        up = block_type.endswith("U")
        attention = block_type.endswith("A")
        decoder_blocks.append(
            DecoderBlock(
                in_channels=in_channel,
                out_channels=out_channel,
                skip_channels=skip_channel,
                up=up,
                attention=attention,
                **kwargs,
            )
        )
        in_channel = out_channel

    return decoder_blocks


class Denoiser(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_block_types: tuple[str] = get_encoder_blocks_types(),
        decoder_block_types: tuple[str] = get_decoder_blocks_types(),
        encoder_out_channels: tuple[int] = get_encoder_out_channels(),
        decoder_out_channels: tuple[int] = get_decoder_out_channels(),
        skip_connections: tuple[bool] = get_skip_connections(),
        dropout_rate: float = 0.0,
        sigma_data: float = 0.5,
        encoder_add_factor: float = 0.3,
        decoder_add_factor: float = 0.3,
        embedding_dim: int = 768,
        num_heads: int = 4,
    ):
        super().__init__()
        assert len(encoder_block_types) == len(
            encoder_out_channels
        ), f"encoder_block_types and encoder_out_channels must have the same length, got {len(encoder_block_types)} and {len(encoder_out_channels)}"
        assert len(decoder_block_types) == len(
            decoder_out_channels
        ), f"decoder_block_types and decoder_out_channels must have the same length, got {len(decoder_block_types)} and {len(decoder_out_channels)}"
        assert len(skip_connections) == len(
            decoder_out_channels
        ), f"skip_connections must have the same length as decoder_out_channels, got {len(skip_connections)} and {len(decoder_out_channels)}"

        (
            encoder_block_types,
            decoder_block_types,
            encoder_out_channels,
            decoder_out_channels,
            skip_connections,
        ) = map(
            tuple,
            (
                encoder_block_types,
                decoder_block_types,
                encoder_out_channels,
                decoder_out_channels,
                skip_connections,
            ),
        )
        self.skip_connections = skip_connections

        self.conv_in = Conv2d(in_channels + 1, encoder_out_channels[0], 3)
        self.conv_out = Conv2d(decoder_out_channels[-1], out_channels, 1)
        self.gain_out = nn.Parameter(torch.zeros(1))

        self.encoder_blocks = build_encoder_blocks(
            encoder_block_types,
            encoder_out_channels,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            add_factor=encoder_add_factor,
            num_heads=num_heads,
        )

        skip_channels = get_skip_channels(
            encoder_out_channels, decoder_out_channels, skip_connections
        )

        self.decoder_blocks = build_decoder_blocks(
            decoder_block_types,
            decoder_out_channels,
            skip_channels,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            add_factor=decoder_add_factor,
            num_heads=num_heads,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder_block_types = encoder_block_types
        self.decoder_block_types = decoder_block_types
        self.encoder_out_channels = encoder_out_channels
        self.decoder_out_channels = decoder_out_channels
        self.skip_connections = skip_connections
        self.dropout_rate = dropout_rate
        self.sigma_data = sigma_data
        self.encoder_add_factor = encoder_add_factor
        self.decoder_add_factor = decoder_add_factor
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

    def forward(self, noisy_image: Tensor, sigma: Tensor, embedding: Tensor):
        if sigma.ndim == 0:
            sigma = sigma * torch.ones(
                noisy_image.shape[0], dtype=noisy_image.dtype, device=noisy_image.device
            )

        sigma = sigma.view(-1, 1, 1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

        # Input block
        x = c_in * noisy_image
        ones_tensor = torch.ones_like(x[:, 0:1, :, :])
        x = torch.cat((x, ones_tensor), dim=1)
        x = self.conv_in(x)

        skips = []
        skips.append(x)
        for block in self.encoder_blocks:
            x = block(x, embedding)
            skips.append(x)

        for block, has_skip in zip(self.decoder_blocks, self.skip_connections):
            if has_skip:
                x = block(x, embedding, skips.pop())
            else:
                x = block(x, embedding)

        # Output block
        denoised_image = self.conv_out(x) * self.gain_out
        denoised_image = denoised_image * c_out + noisy_image * c_skip

        return denoised_image


class DenoiserWrapper(nn.Module):
    """
    A denoiser proposed in [1]. It wraps a neural network with a skip-connection-like structure.

    Parameters:
        net: The neural network.
        sigma_data: The estimated standard deviation of the data.

    Returns:
        The denoised image.

    [1]: Karras T, Aittala M, Aila T, et al. Elucidating the design space of diffusion-based generative models[J].
         Advances in Neural Information Processing Systems, 2022, 35: 26565-26577.


    """

    def __init__(self, net: nn.Module, sigma_data: float):
        super().__init__()

        self.net = net
        self._sigma_data = sigma_data

    @property
    def sigma_data(self) -> float:
        return self._sigma_data

    def forward(
        self, noisy_image: Tensor, sigma: Tensor, embedding: Tensor | None = None
    ) -> Tensor:
        if sigma.ndim == 0:
            sigma = sigma * torch.ones(
                noisy_image.shape[0], dtype=noisy_image.dtype, device=noisy_image.device
            )
        sigma = sigma.view(-1, 1, 1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F = self.net(c_in * noisy_image, c_noise.flatten(), embedding)
        D = c_skip * noisy_image + c_out * F
        return D
