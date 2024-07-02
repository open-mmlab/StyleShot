import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x

class Low_CNN(nn.Module):
    def __init__(self, cin=192, ksize=1, sk=False, use_conv=True):
        super(Low_CNN, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.body  = nn.Sequential(ResnetBlock(320, 320, down=False, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(320, 640, down=False, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(640, 1280, down=True, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(1280, 1280, down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.conv_in = nn.Conv2d(cin, 320, 3, 1, 1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adapter = nn.Linear(1280, 1280)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv_in(x)
        x = self.body(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.adapter(x)
        return x


class Middle_CNN(nn.Module):
    def __init__(self, cin=192, ksize=1, sk=False, use_conv=True):
        super(Middle_CNN, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.body  = nn.Sequential(ResnetBlock(320, 320, down=False, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(320, 640, down=False, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(640, 640, down=True, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(640, 1280, down=True, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(1280, 1280, down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.conv_in = nn.Conv2d(cin, 320, 3, 1, 1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adapter = nn.Linear(1280, 1280)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv_in(x)
        x = self.body(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.adapter(x)
        return x


class High_CNN(nn.Module):
    def __init__(self, cin=192, ksize=1, sk=False, use_conv=True):
        super(High_CNN, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.body  = nn.Sequential(ResnetBlock(320, 320, down=False, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(320, 640, down=False, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(640, 640, down=True, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(640, 640, down=True, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(640, 1280, down=True, ksize=ksize, sk=sk, use_conv=use_conv),
                                   ResnetBlock(1280, 1280, down=False, ksize=ksize, sk=sk, use_conv=use_conv))
        self.conv_in = nn.Conv2d(cin, 320, 3, 1, 1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adapter = nn.Linear(1280, 1280)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv_in(x)
        x = self.body(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.adapter(x)
        return x


class Style_Aware_Encoder(torch.nn.Module):
    def __init__(self, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.projection_dim = self.image_encoder.config.projection_dim
        self.num_positions = 59
        self.embed_dim = 1280
        self.cnn = nn.ModuleList(
            [High_CNN(sk=True, use_conv=False),
            Middle_CNN(sk=True, use_conv=False),
            Low_CNN(sk=True, use_conv=False)]
        )
        self.style_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(self.embed_dim)),
            nn.Parameter(torch.randn(self.embed_dim)),
            nn.Parameter(torch.randn(self.embed_dim))]
        )
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, inputs, batch_size=1):
        embeddings = []
        for idx, x in enumerate(inputs):
            class_embed = self.style_embeddings[idx].expand(batch_size, 1, -1)
            patch_embed = self.cnn[idx](x)
            patch_embed = patch_embed.view(batch_size, -1, patch_embed.shape[1])
            embedding = torch.cat([class_embed, patch_embed], dim=1)
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)  # [B, 256, 1280] - [B, P, 1280]
        embeddings = self.image_encoder.vision_model.pre_layrnorm(embeddings)
        encoder_outputs = self.image_encoder.vision_model.encoder(
            inputs_embeds=embeddings,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, [0, 9, 26], :]
        pooled_output = self.image_encoder.vision_model.post_layernorm(pooled_output)
        out = self.image_encoder.visual_projection(pooled_output)
        return out
