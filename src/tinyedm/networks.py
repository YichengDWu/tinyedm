import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = np.sqrt(n.numel() / x.numel())
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
        w = normalize(self.weight) / np.sqrt(fan_in)
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
        w = normalize(self.weight) / np.sqrt(fan_in)
        x = F.linear(x, w)
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}"


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


def pixel_norm(x: Tensor, eps: float = 1e-4, dim=1) -> Tensor:
    return x / (torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + eps))


def mp_silu(x: Tensor) -> Tensor:
    return F.silu(x) / 0.596


def mp_add(a: Tensor, b: Tensor, t: float = 0.3) -> Tensor:
    scale = np.sqrt(t**2 + (1 - t) ** 2)
    return ((1 - t) * a + t * b) / scale


def mp_cat(a: Tensor, b: Tensor, t: float = 0.5) -> Tensor:
    N_a, N_b = a[0].numel(), b[0].numel()
    scale = np.sqrt((N_a + N_b) / (t**2 + (1 - t) ** 2))
    out = torch.cat([(1 - t) / np.sqrt(N_a) * a, t / np.sqrt(N_b) * b], dim=1)
    return out * scale


class ClassEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.linear = Linear(num_embeddings, embedding_dim)

    def forward(self, class_idx: Tensor):
        class_emb = F.one_hot(class_idx, self.num_embeddings).to(
            dtype=self.linear.dtype, device=self.linear.device
        )
        return self.linear(class_emb * np.sqrt(self.num_embeddings))


class FourierFeatures(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.register_buffer("freqs", torch.randn(embedding_dim))
        self.register_buffer("phases", torch.rand(embedding_dim))

    def forward(self, x):
        x = torch.outer(x.flatten(), self.freqs) + self.phases
        x = torch.cos(2 * torch.pi * x) * np.sqrt(2)
        return x


class CosineAttention(nn.Module):
    def __init__(self, embed_dimension: int, head_dim: int = 64):
        super().__init__()
        assert embed_dimension % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = embed_dimension // head_dim
        self.embed_dimension = embed_dimension
        self.c_attn = Conv2d(embed_dimension, 3 * embed_dimension, 1)
        self.c_proj = Conv2d(embed_dimension, embed_dimension, 1)

    def forward(self, x):
        input = x
        b, c, h, w = x.shape
        x_proj = self.c_attn(x)  # (b, c, h, w) -> (b, 3*c, h, w)
        x_proj = x_proj.view(b, -1, h * w).transpose(1, 2)  # (b, h*w, 3*c)

        q, k, v = x_proj.chunk(3, -1)
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (b, num_heads, h*w, head_dim)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = pixel_norm(q), pixel_norm(k), pixel_norm(v)

        res = F.scaled_dot_product_attention(q, k, v)  # (b, num_heads, h*w, head_dim)

        res = res.transpose(-1, -2).reshape(b, -1, h, w)
        res = self.c_proj(res)

        out = mp_add(input, res)
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        down: bool,
        attention: bool,
        head_dim: int = 64,
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
            CosineAttention(out_channels, head_dim) if attention else nn.Identity()
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

        res = res * (self.embed(embedding) * self.gain + 1).unsqueeze(-1).unsqueeze(-1)
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
        head_dim: int = 64,
        skip_channels: int = 0,
        dropout_rate: float = 0.0,
        add_factor: float = 0.3,
        cat_factor: float = 0.5,
    ):
        super().__init__()

        self.add_factor = add_factor
        self.cat_factor = cat_factor

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
            CosineAttention(out_channels, head_dim) if attention else nn.Identity()
        )

        # embedding layer
        self.embed = Linear(embedding_dim, out_channels)
        self.gain = nn.Parameter(torch.ones(1))

    def forward(
        self, input: Tensor, embedding: Tensor, skip: Tensor | None = None
    ) -> Tensor:
        if skip is not None:
            input = mp_cat(input, skip, self.cat_factor)
        x = self.resample(input)
        res = x
        x = self.conv_1x1(x)

        res = mp_silu(res)
        res = self.conv_3x3_1(res)

        res = res * (self.embed(embedding) * self.gain + 1).unsqueeze(-1).unsqueeze(-1)
        res = mp_silu(res)
        res = self.dropout(res)
        res = self.conv_3x3_2(res)

        out = mp_add(x, res, self.add_factor)
        out = self.attention(out)
        return out


class Embedding(nn.Module):
    def __init__(
        self,
        fourier_dim: int,
        output_dim: int,
        num_class_embeds: int | None = None,
        add_factor: float = 0.5,
    ):
        super().__init__()
        self.add_factor = add_factor
        self.fourier_features = FourierFeatures(fourier_dim)
        self.sigma_embed = Linear(fourier_dim, output_dim)
        self.class_embed = None
        if num_class_embeds is not None:
            self.class_embed = ClassEmbedding(num_class_embeds, output_dim)

    def forward(self, sigmas, class_labels=None):
        c_noise = sigmas.log() / 4
        embedding = self.fourier_features(c_noise)
        embedding = self.sigma_embed(embedding)

        if self.class_embed is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when doing class conditioning"
                )

            class_emb = self.class_embed(class_labels)
            embedding = mp_add(embedding, class_emb, self.add_factor)
        out = mp_silu(embedding)
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
    return (2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20)


def get_skip_channels(
    encoder_out_channels: tuple[int],
    decoder_out_channels: tuple[int],
    skip_connections: tuple[int],
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


class UNet2DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        fourier_dim: int = 192,
        encoder_block_types: tuple[str] = get_encoder_blocks_types(),
        decoder_block_types: tuple[str] = get_decoder_blocks_types(),
        encoder_out_channels: tuple[int] = get_encoder_out_channels(),
        decoder_out_channels: tuple[int] = get_decoder_out_channels(),
        skip_connections: tuple[int] = get_skip_connections(),
        dropout_rate: float = 0.0,
        num_class_embeds: int | None = None,
        embedding_add_factor: float = 0.5,
        encoder_add_factor: float = 0.3,
        decoder_add_factor: float = 0.3,
        decoder_cat_factor: float = 0.5,
        embedding_dim: int = 768,
        head_dim: int = 64,
    ):
        super().__init__()
        assert len(encoder_block_types) == len(
            encoder_out_channels
        ), "encoder_block_typs and encoder_out_channels must have the same length"
        assert len(decoder_block_types) == len(
            decoder_out_channels
        ), "decoder_block_types and decoder_out_channels must have the same length"
        assert (
            len(skip_connections) == len(encoder_out_channels) + 1
        ), "skip_connections must have the same length as encoder_out_channels + 1"

        self.sigma_data = 0.5
        self.skip_connections = skip_connections
        self.embed = Embedding(
            fourier_dim,
            embedding_dim,
            num_class_embeds=num_class_embeds,
            add_factor=embedding_add_factor,
        )

        self.conv_in = Conv2d(in_channels + 1, encoder_out_channels[0], 3)
        self.conv_out = Conv2d(decoder_out_channels[-1], out_channels, 1)
        self.u = Linear(embedding_dim, 1)
        self.gain = nn.Parameter(torch.ones(1))

        self.encoder_blocks = build_encoder_blocks(
            encoder_block_types,
            encoder_out_channels,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
            add_factor=encoder_add_factor,
            head_dim=head_dim,
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
            cat_factor=decoder_cat_factor,
            head_dim=head_dim,
        )

    @property
    def embedding_dim(self):
        return self.embedding_dim

    def forward(
        self, noisy_image: Tensor, sigma: Tensor, class_label: Tensor | None = None
    ):
        if sigma.ndim == 0:
            sigma = sigma * torch.ones(
                noisy_image.shape[0], dtype=noisy_image.dtype, device=noisy_image.device
            )

        # Embedding block
        embedding = self.embed(
            sigma, class_label
        )  # c_noise is already calculated in the embedding block

        uncertainty = self.u(embedding)

        sigma = sigma.view(-1, 1, 1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()

        # Input block
        x = c_in * noisy_image
        ones_tensor = torch.ones_like(x[:, 0:1, :, :])
        x = torch.cat((x, ones_tensor), dim=1)
        x = self.conv_in(x)

        skips = [
            x,
        ]
        for block in self.encoder_blocks:
            x = block(x, embedding)
            skips.append(x)

        j = len(self.skip_connections) - 1
        for i, block in enumerate(self.decoder_blocks):
            if i in self.skip_connections:
                x = block(x, embedding, skips[j])
                j -= 1
            else:
                x = block(x, embedding)

        # Output block
        denoised_image = self.conv_out(x) * self.gain
        denoised_image = denoised_image * c_out + noisy_image * c_skip

        return denoised_image, uncertainty
