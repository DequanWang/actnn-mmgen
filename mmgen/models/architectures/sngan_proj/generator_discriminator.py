# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_activation_layer, constant_init,
                      xavier_init)
from mmcv.runner import load_checkpoint
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix
from mmcv.utils import is_list_of
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm

from mmgen.models.builder import MODULES, build_module
from mmgen.utils import check_dist_init
from mmgen.utils.logger import get_root_logger
from ..common import get_module_device


@MODULES.register_module('SAGANGenerator')
@MODULES.register_module()
class SNGANGenerator(nn.Module):
    r"""Generator for SNGAN / Proj-GAN. The implementation refers to
    https://github.com/pfnet-research/sngan_projection/tree/master/gen_models

    In our implementation, we have two notable design. Namely,
    ``channels_cfg`` and ``blocks_cfg``.

    ``channels_cfg``: In default config of SNGAN / Proj-GAN, the number of
        ResBlocks and the channels of those blocks are corresponding to the
        resolution of the output image. Therefore, we allow user to define
        ``channels_cfg`` to try their own models. We also provide a default
        config to allow users to build the model only from the output
        resolution.

    ``block_cfg``: In reference code, the generator consists of a group of
        ResBlock. However, in our implementation, to make this model more
        generalize, we support defining ``blocks_cfg`` by users and loading
        the blocks by calling the build_module method.

    Args:
        output_scale (int): Output scale for the generated image.
        num_classes (int, optional): The number classes you would like to
            generate. This arguments would influence the structure of the
            intermedia blocks and label sampling operation in ``forward``
            (e.g. If num_classes=0, ConditionalNormalization layers would
            degrade to unconditional ones.). This arguments would be passed
            to intermedia blocks by overwrite their config. Defaults to 0.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Default to 64.
        out_channels (int, optional): Channels of the output images.
            Default to 3.
        input_scale (int, optional): Input scale for the features.
            Defaults to 4.
        noise_size (int, optional): Size of the input noise vector.
            Default to 128.
        attention_cfg (dict, optional): Config for the self-attention block.
            Default to ``dict(type='SelfAttentionBlock')``.
        attention_after_nth_block (int | list[int], optional): Self attention
            block would be added after which *ConvBlock*. If ``int`` is passed,
            only one attention block would be added. If ``list`` is passed,
            self-attention blocks would be added after multiple ConvBlocks.
            To be noted that if the input is smaller than ``1``,
            self-attention corresponding to this index would be ignored.
            Default to 0.
        channels_cfg (list | dict[list], optional): Config for input channels
            of the intermedia blocks. If list is passed, each element of the
            list means the input channels of current block is how many times
            compared to the ``base_channels``. For block ``i``, the input and
            output channels should be ``channels_cfg[i]`` and
            ``channels_cfg[i+1]`` If dict is provided, the key of the dict
            should be the output scale and corresponding value should be a list
            to define channels.  Default: Please refer to
            ``_defualt_channels_cfg``.
        blocks_cfg (dict, optional): Config for the intermedia blocks.
            Defaults to ``dict(type='SNGANGenResBlock')``
        act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='ReLU')``.
        use_cbn (bool, optional): Whether use conditional normalization. This
            argument would pass to norm layers. Defaults to True.
        auto_sync_bn (bool, optional): Whether convert Batch Norm to
            Synchronized ones when Distributed training is on. Defaults to
            True.
        with_spectral_norm (bool, optional): Whether use spectral norm for
            conv blocks or not. Default to False.
        with_embedding_spectral_norm (bool, optional): Whether use spectral
            norm for embedding layers in normalization blocks or not. If not
            specified (set as ``None``), ``with_embedding_spectral_norm`` would
            be set as the same value as ``with_spectral_norm``.
            Defaults to None.
        norm_eps (float, optional): eps for Normalization layers (both
            conditional and non-conditional ones). Default to `1e-4`.
        sn_eps (float, optional): eps for spectral normalization operation.
            Defaults to `1e-12`.
        init_cfg (string, optional): Config for weight initialization.
            Defaults to ``dict(type='BigGAN')``.
        pretrained (str | dict, optional): Path for the pretrained model or
            dict containing information for pretained models whose necessary
            key is 'ckpt_path'. Besides, you can also provide 'prefix' to load
            the generator part from the whole state dict.  Defaults to None.
    """

    # default channel factors
    _defualt_channels_cfg = {
        32: [1, 1, 1],
        64: [16, 8, 4, 2],
        128: [16, 16, 8, 4, 2]
    }

    def __init__(self,
                 output_scale,
                 num_classes=0,
                 base_channels=64,
                 out_channels=3,
                 input_scale=4,
                 noise_size=128,
                 attention_cfg=dict(type='SelfAttentionBlock'),
                 attention_after_nth_block=0,
                 channels_cfg=None,
                 blocks_cfg=dict(type='SNGANGenResBlock'),
                 act_cfg=dict(type='ReLU'),
                 use_cbn=True,
                 auto_sync_bn=True,
                 with_spectral_norm=False,
                 with_embedding_spectral_norm=None,
                 norm_eps=1e-4,
                 sn_eps=1e-12,
                 init_cfg=dict(type='BigGAN'),
                 pretrained=None):

        super().__init__()

        self.input_scale = input_scale
        self.output_scale = output_scale
        self.noise_size = noise_size
        self.num_classes = num_classes
        self.init_type = init_cfg.get('type', None)

        self.blocks_cfg = deepcopy(blocks_cfg)

        self.blocks_cfg.setdefault('num_classes', num_classes)
        self.blocks_cfg.setdefault('act_cfg', act_cfg)
        self.blocks_cfg.setdefault('use_cbn', use_cbn)
        self.blocks_cfg.setdefault('auto_sync_bn', auto_sync_bn)
        self.blocks_cfg.setdefault('with_spectral_norm', with_spectral_norm)

        # set `norm_spectral_norm` as `with_spectral_norm` if not defined
        with_embedding_spectral_norm = with_embedding_spectral_norm \
            if with_embedding_spectral_norm is not None else with_spectral_norm
        self.blocks_cfg.setdefault('with_embedding_spectral_norm',
                                   with_embedding_spectral_norm)
        self.blocks_cfg.setdefault('init_cfg', init_cfg)
        self.blocks_cfg.setdefault('norm_eps', norm_eps)
        self.blocks_cfg.setdefault('sn_eps', sn_eps)

        channels_cfg = deepcopy(self._defualt_channels_cfg) \
            if channels_cfg is None else deepcopy(channels_cfg)
        if isinstance(channels_cfg, dict):
            if output_scale not in channels_cfg:
                raise KeyError(f'`output_scale={output_scale} is not found in '
                               '`channel_cfg`, only support configs for '
                               f'{[chn for chn in channels_cfg.keys()]}')
            self.channel_factor_list = channels_cfg[output_scale]
        elif isinstance(channels_cfg, list):
            self.channel_factor_list = channels_cfg
        else:
            raise ValueError('Only support list or dict for `channel_cfg`, '
                             f'receive {type(channels_cfg)}')

        self.noise2feat = nn.Linear(
            noise_size,
            input_scale**2 * base_channels * self.channel_factor_list[0])
        if with_spectral_norm:
            self.noise2feat = spectral_norm(self.noise2feat)

        # check `attention_after_nth_block`
        if not isinstance(attention_after_nth_block, list):
            attention_after_nth_block = [attention_after_nth_block]
        if not is_list_of(attention_after_nth_block, int):
            raise ValueError('`attention_after_nth_block` only support int or '
                             'a list of int. Please check your input type.')

        self.conv_blocks = nn.ModuleList()
        self.attention_block_idx = []
        for idx in range(len(self.channel_factor_list)):
            factor_input = self.channel_factor_list[idx]
            factor_output = self.channel_factor_list[idx+1] \
                if idx < len(self.channel_factor_list)-1 else 1

            # get block-specific config
            block_cfg_ = deepcopy(self.blocks_cfg)
            block_cfg_['in_channels'] = factor_input * base_channels
            block_cfg_['out_channels'] = factor_output * base_channels
            self.conv_blocks.append(build_module(block_cfg_))

            # build self-attention block
            # `idx` is start from 0, add 1 to get the index
            if idx + 1 in attention_after_nth_block:
                self.attention_block_idx.append(len(self.conv_blocks))
                attn_cfg_ = deepcopy(attention_cfg)
                attn_cfg_['in_channels'] = factor_output * base_channels
                self.conv_blocks.append(build_module(attn_cfg_))

        to_rgb_norm_cfg = dict(type='BN', eps=norm_eps)
        if check_dist_init() and auto_sync_bn:
            to_rgb_norm_cfg['type'] = 'SyncBN'

        self.to_rgb = ConvModule(
            factor_output * base_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            norm_cfg=to_rgb_norm_cfg,
            act_cfg=act_cfg,
            order=('norm', 'act', 'conv'),
            with_spectral_norm=with_spectral_norm)
        self.final_act = build_activation_layer(dict(type='Tanh'))

        self.init_weights(pretrained)

    def forward(self, noise, num_batches=0, label=None, return_noise=False):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            label (torch.Tensor | callable | None): You can directly give a
                batch of label through a ``torch.Tensor`` or offer a callable
                function to sample a batch of label data. Otherwise, the
                ``None`` indicates to use the default label sampler.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output
                image will be returned. Otherwise, a dict contains
                ``fake_image``, ``noise_batch`` and ``label_batch``
                would be returned.
        """
        if isinstance(noise, torch.Tensor):
            assert noise.shape[1] == self.noise_size
            assert noise.ndim == 2, ('The noise should be in shape of (n, c), '
                                     f'but got {noise.shape}')
            noise_batch = noise
        # receive a noise generator and sample noise.
        elif callable(noise):
            noise_generator = noise
            assert num_batches > 0
            noise_batch = noise_generator((num_batches, self.noise_size))
        # otherwise, we will adopt default noise sampler.
        else:
            assert num_batches > 0
            noise_batch = torch.randn((num_batches, self.noise_size))

        if isinstance(label, torch.Tensor):
            assert label.ndim == 1, ('The label shoube be in shape of (n, )'
                                     f'but got {label.shape}.')
            label_batch = label
        elif callable(label):
            label_generator = label
            assert num_batches > 0
            label_batch = label_generator(num_batches)
        elif self.num_classes == 0:
            label_batch = None
        else:
            assert num_batches > 0
            label_batch = torch.randint(0, self.num_classes, (num_batches, ))

        # dirty code for putting data on the right device
        noise_batch = noise_batch.to(get_module_device(self))
        if label_batch is not None:
            label_batch = label_batch.to(get_module_device(self))

        x = self.noise2feat(noise_batch)
        x = x.reshape(x.size(0), -1, self.input_scale, self.input_scale)

        for idx, conv_block in enumerate(self.conv_blocks):
            if idx in self.attention_block_idx:
                x = conv_block(x)
            else:
                x = conv_block(x, label_batch)

        out_feat = self.to_rgb(x)
        out_img = self.final_act(out_feat)

        if return_noise:
            return dict(
                fake_img=out_img, noise_batch=noise_batch, label=label_batch)
        return out_img

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for SNGAN-Proj and SAGAN. If ``pretrained=None``,
        weight initialization would follow the ``INIT_TYPE`` in
        ``init_cfg=dict(type=INIT_TYPE)``.

        For SNGAN-Proj,
        (``INIT_TYPE.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']``),
        we follow the initialization method in the official Chainer's
        implementation (https://github.com/pfnet-research/sngan_projection).

        For SAGAN (``INIT_TYPE.upper() == 'SAGAN'``), we follow the
        initialization method in official tensorflow's implementation
        (https://github.com/brain-research/self-attention-gan).

        Besides the reimplementation of the official code's initialization, we
        provide BigGAN's and Pytorch-StudioGAN's style initialization
        (``INIT_TYPE.upper() == BIGGAN`` and ``INIT_TYPE.upper() == STUDIO``).
        Please refer to https://github.com/ajbrock/BigGAN-PyTorch and
        https://github.com/POSTECH-CVLab/PyTorch-StudioGAN.

        Args:
            pretrained (str | dict, optional): Path for the pretrained model or
                dict containing information for pretained models whose
                necessary key is 'ckpt_path'. Besides, you can also provide
                'prefix' to load the generator part from the whole state dict.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif isinstance(pretrained, dict):
            ckpt_path = pretrained.get('ckpt_path', None)
            assert ckpt_path is not None
            prefix = pretrained.get('prefix', '')
            map_location = pretrained.get('map_location', 'cpu')
            strict = pretrained.get('strict', True)
            state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                      map_location)
            self.load_state_dict(state_dict, strict=strict)
        elif pretrained is None:
            if self.init_type.upper() in 'STUDIO':
                # initialization method from Pytorch-StudioGAN
                #   * weight: orthogonal_init gain=1
                #   * bias  : 0
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        nn.init.orthogonal_(m.weight)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.data.fill_(0.)
            elif self.init_type.upper() == 'BIGGAN':
                # initialization method from BigGAN-pytorch
                #   * weight: xavier_init gain=1
                #   * bias  : default
                for n, m in self.named_modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        xavier_uniform_(m.weight, gain=1)
            elif self.init_type.upper() == 'SAGAN':
                # initialization method from official tensorflow code
                #   * weight          : xavier_init gain=1
                #   * bias            : 0
                #   * weight_embedding: 1
                #   * bias_embedding  : 0
                for n, m in self.named_modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        xavier_init(m, gain=1, distribution='uniform')
                    if isinstance(m, nn.Embedding):
                        # To be noted that here we initialize the embedding
                        # layer in cBN with specific prefix. If you implement
                        # your own cBN and want to use this initialization
                        # method, please make sure the embedding layers in
                        # your implementation have the same prefix as ours.
                        if 'weight' in n:
                            constant_init(m, 1)
                        if 'bias' in n:
                            constant_init(m, 0)
            elif self.init_type.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']:
                # initialization method from the official chainer code
                #   * conv.weight     : xavier_init gain=sqrt(2)
                #   * shortcut.weight : xavier_init gain=1
                #   * bias            : 0
                #   * weight_embedding: 1
                #   * bias_embedding  : 0
                for n, m in self.named_modules():
                    if isinstance(m, nn.Conv2d):
                        if 'shortcut' in n or 'to_rgb' in n:
                            xavier_init(m, gain=1, distribution='uniform')
                        else:
                            xavier_init(
                                m, gain=np.sqrt(2), distribution='uniform')
                    if isinstance(m, nn.Linear):
                        xavier_init(m, gain=1, distribution='uniform')
                    if isinstance(m, nn.Embedding):
                        # To be noted that here we initialize the embedding
                        # layer in cBN with specific prefix. If you implement
                        # your own cBN and want to use this initialization
                        # method, please make sure the embedding layers in
                        # your implementation have the same prefix as ours.
                        if 'weight' in n:
                            constant_init(m, 1)
                        if 'bias' in n:
                            constant_init(m, 0)
            else:
                raise NotImplementedError('Unknown initialization method: '
                                          f'\'{self.init_type}\'')

        else:
            raise TypeError("'pretrined' must be a str or None. "
                            f'But receive {type(pretrained)}.')


@MODULES.register_module('SAGANDiscriminator')
@MODULES.register_module()
class ProjDiscriminator(nn.Module):
    r"""Discriminator for SNGAN / Proj-GAN. The implementation is refer to
    https://github.com/pfnet-research/sngan_projection/tree/master/dis_models

    The overall structure of the projection discriminator can be split into a
    ``from_rgb`` layer, a group of ResBlocks, a linear decision layer, and a
    projection layer. To support defining custom layers, we introduce
    ``from_rgb_cfg`` and ``blocks_cfg``.

    The design of the model structure is highly corresponding to the output
    resolution. Therefore, we provide `channels_cfg` and `downsample_cfg` to
    control the input channels and the downsample behavior of the intermedia
    blocks.

    ``downsample_cfg``: In default config of SNGAN / Proj-GAN, whether to apply
        downsample in each intermedia blocks is quite flexible and
        corresponding to the resolution of the output image. Therefore, we
        support user to define the ``downsample_cfg`` by themselves, and to
        control the structure of the discriminator.

    ``channels_cfg``: In default config of SNGAN / Proj-GAN, the number of
        ResBlocks and the channels of those blocks are corresponding to the
        resolution of the output image. Therefore, we allow user to define
        `channels_cfg` for try their own models.  We also provide a default
        config to allow users to build the model only from the output
        resolution.

    Args:
        input_scale (int): The scale of the input image.
        num_classes (int, optional): The number classes you would like to
            generate. If num_classes=0, no label projection would be used.
            Default to 0.
        base_channels (int, optional): The basic channel number of the
            discriminator. The other layers contains channels based on this
            number.  Defaults to 128.
        input_channels (int, optional): Channels of the input image.
            Defaults to 3.
        attention_cfg (dict, optional): Config for the self-attention block.
            Default to ``dict(type='SelfAttentionBlock')``.
        attention_after_nth_block (int | list[int], optional): Self-attention
            block would be added after which *ConvBlock* (including the head
            block). If ``int`` is passed, only one attention block would be
            added. If ``list`` is passed, self-attention blocks would be added
            after multiple ConvBlocks. To be noted that if the input is
            smaller than ``1``, self-attention corresponding to this index
            would be ignored. Default to 0.
        channels_cfg (list | dict[list], optional): Config for input channels
            of the intermedia blocks. If list is passed, each element of the
            list means the input channels of current block is how many times
            compared to the ``base_channels``. For block ``i``, the input and
            output channels should be ``channels_cfg[i]`` and
            ``channels_cfg[i+1]`` If dict is provided, the key of the dict
            should be the output scale and corresponding value should be a list
            to define channels.  Default: Please refer to
            ``_defualt_channels_cfg``.
        downsample_cfg (list[bool] | dict[list], optional): Config for
            downsample behavior of the intermedia layers. If a list is passed,
            ``downsample_cfg[idx] == True`` means apply downsample in idx-th
            block, and vice versa. If dict is provided, the key dict should
            be the input scale of the image and corresponding value should be
            a list ti define the downsample behavior. Default: Please refer
            to ``_default_downsample_cfg``.
        from_rgb_cfg (dict, optional): Config for the first layer to convert
            rgb image to feature map. Defaults to
            ``dict(type='SNGANDiscHeadResBlock')``.
        blocks_cfg (dict, optional): Config for the intermedia blocks.
            Defaults to ``dict(type='SNGANDiscResBlock')``
        act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='ReLU')``.
        with_spectral_norm (bool, optional): Whether use spectral norm for
            all conv blocks or not. Default to True.
        init_cfg (dict, optional): Config for weight initialization.
            Default to ``dict(type='BigGAN')``.
        pretrained (str | dict , optional): Path for the pretrained model or
            dict containing information for pretained models whose necessary
            key is 'ckpt_path'. Besides, you can also provide 'prefix' to load
            the generator part from the whole state dict.  Defaults to None.
    """

    # default channel factors
    _defualt_channels_cfg = {
        32: [1, 1, 1],
        64: [2, 4, 8, 16],
        128: [2, 4, 8, 16, 16],
    }

    # default downsample behavior
    _defualt_downsample_cfg = {
        32: [True, False, False],
        64: [True, True, True, True],
        128: [True, True, True, True, False]
    }

    def __init__(self,
                 input_scale,
                 num_classes=0,
                 base_channels=128,
                 input_channels=3,
                 attention_cfg=dict(type='SelfAttentionBlock'),
                 attention_after_nth_block=-1,
                 channels_cfg=None,
                 downsample_cfg=None,
                 from_rgb_cfg=dict(type='SNGANDiscHeadResBlock'),
                 blocks_cfg=dict(type='SNGANDiscResBlock'),
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=True,
                 sn_eps=1e-12,
                 init_cfg=dict(type='BigGAN'),
                 pretrained=None):

        super().__init__()

        self.init_type = init_cfg.get('type', None)

        # add SN options and activation function options to cfg
        self.from_rgb_cfg = deepcopy(from_rgb_cfg)
        self.from_rgb_cfg.setdefault('act_cfg', act_cfg)
        self.from_rgb_cfg.setdefault('with_spectral_norm', with_spectral_norm)
        self.from_rgb_cfg.setdefault('init_cfg', init_cfg)

        # add SN options and activation function options to cfg
        self.blocks_cfg = deepcopy(blocks_cfg)
        self.blocks_cfg.setdefault('act_cfg', act_cfg)
        self.blocks_cfg.setdefault('with_spectral_norm', with_spectral_norm)
        self.blocks_cfg.setdefault('sn_eps', sn_eps)
        self.blocks_cfg.setdefault('init_cfg', init_cfg)

        channels_cfg = deepcopy(self._defualt_channels_cfg) \
            if channels_cfg is None else deepcopy(channels_cfg)
        if isinstance(channels_cfg, dict):
            if input_scale not in channels_cfg:
                raise KeyError(f'`input_scale={input_scale} is not found in '
                               '`channel_cfg`, only support configs for '
                               f'{[chn for chn in channels_cfg.keys()]}')
            self.channel_factor_list = channels_cfg[input_scale]
        elif isinstance(channels_cfg, list):
            self.channel_factor_list = channels_cfg
        else:
            raise ValueError('Only support list or dict for `channel_cfg`, '
                             f'receive {type(channels_cfg)}')

        downsample_cfg = deepcopy(self._defualt_downsample_cfg) \
            if downsample_cfg is None else deepcopy(downsample_cfg)
        if isinstance(downsample_cfg, dict):
            if input_scale not in downsample_cfg:
                raise KeyError(f'`output_scale={input_scale} is not found in '
                               '`downsample_cfg`, only support configs for '
                               f'{[chn for chn in downsample_cfg.keys()]}')
            self.downsample_list = downsample_cfg[input_scale]
        elif isinstance(downsample_cfg, list):
            self.downsample_list = downsample_cfg
        else:
            raise ValueError('Only support list or dict for `channel_cfg`, '
                             f'receive {type(downsample_cfg)}')

        if len(self.downsample_list) != len(self.channel_factor_list):
            raise ValueError('`downsample_cfg` should have same length with '
                             '`channels_cfg`, but receive '
                             f'{len(self.downsample_list)} and '
                             f'{len(self.channel_factor_list)}.')

        # check `attention_after_nth_block`
        if not isinstance(attention_after_nth_block, list):
            attention_after_nth_block = [attention_after_nth_block]
        if not all([isinstance(idx, int)
                    for idx in attention_after_nth_block]):
            raise ValueError('`attention_after_nth_block` only support int or '
                             'a list of int. Please check your input type.')

        self.from_rgb = build_module(
            self.from_rgb_cfg,
            dict(in_channels=input_channels, out_channels=base_channels))

        self.conv_blocks = nn.ModuleList()
        # add self-attention block after the first block
        if 1 in attention_after_nth_block:
            attn_cfg_ = deepcopy(attention_cfg)
            attn_cfg_['in_channels'] = base_channels
            self.conv_blocks.append(build_module(attn_cfg_))

        for idx in range(len(self.downsample_list)):
            factor_input = 1 if idx == 0 else self.channel_factor_list[idx - 1]
            factor_output = self.channel_factor_list[idx]

            # get block-specific config
            block_cfg_ = deepcopy(self.blocks_cfg)
            block_cfg_['downsample'] = self.downsample_list[idx]
            block_cfg_['in_channels'] = factor_input * base_channels
            block_cfg_['out_channels'] = factor_output * base_channels
            self.conv_blocks.append(build_module(block_cfg_))

            # build self-attention block
            # the first ConvBlock is `from_rgb` block,
            # add 2 to get the index of the ConvBlocks
            if idx + 2 in attention_after_nth_block:
                attn_cfg_ = deepcopy(attention_cfg)
                attn_cfg_['in_channels'] = factor_output * base_channels
                self.conv_blocks.append(build_module(attn_cfg_))

        self.decision = nn.Linear(factor_output * base_channels, 1)

        if with_spectral_norm:
            self.decision = spectral_norm(self.decision)

        self.num_classes = num_classes

        # In this case, discriminator is designed for conditional synthesis.
        if num_classes > 0:
            self.proj_y = nn.Embedding(num_classes,
                                       factor_output * base_channels)
            if with_spectral_norm:
                self.proj_y = spectral_norm(self.proj_y)

        self.activate = build_activation_layer(act_cfg)
        self.init_weights(pretrained)

    def forward(self, x, label=None):
        """Forward function. If `self.num_classes` is larger than 0, label
        projection would be used.

        Args:
            x (torch.Tensor): Fake or real image tensor.
            label (torch.Tensor, options): Label correspond to the input image.
                Noted that, if `self.num_classed` is larger than 0,
                `label` should not be None.  Default to None.

        Returns:
            torch.Tensor: Prediction for the reality of the input image.
        """
        h = self.from_rgb(x)
        for conv_block in self.conv_blocks:
            h = conv_block(h)
        h = self.activate(h)
        h = torch.sum(h, dim=[2, 3])
        out = self.decision(h)

        if self.num_classes > 0:
            w_y = self.proj_y(label)
            out = out + torch.sum(w_y * h, dim=1, keepdim=True)
        return out.view(out.size(0), -1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for SNGAN-Proj and SAGAN. If ``pretrained=None`` and
        weight initialization would follow the ``INIT_TYPE`` in
        ``init_cfg=dict(type=INIT_TYPE)``.

        For SNGAN-Proj
        (``INIT_TYPE.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']``),
        we follow the initialization method in the official Chainer's
        implementation (https://github.com/pfnet-research/sngan_projection).

        For SAGAN (``INIT_TYPE.upper() == 'SAGAN'``), we follow the
        initialization method in official tensorflow's implementation
        (https://github.com/brain-research/self-attention-gan).

        Besides the reimplementation of the official code's initialization, we
        provide BigGAN's and Pytorch-StudioGAN's style initialization
        (``INIT_TYPE.upper() == BIGGAN`` and ``INIT_TYPE.upper() == STUDIO``).
        Please refer to https://github.com/ajbrock/BigGAN-PyTorch and
        https://github.com/POSTECH-CVLab/PyTorch-StudioGAN.

        Args:
            pretrained (str | dict, optional): Path for the pretrained model or
                dict containing information for pretained models whose
                necessary key is 'ckpt_path'. Besides, you can also provide
                'prefix' to load the generator part from the whole state dict.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif isinstance(pretrained, dict):
            ckpt_path = pretrained.get('ckpt_path', None)
            assert ckpt_path is not None
            prefix = pretrained.get('prefix', '')
            map_location = pretrained.get('map_location', 'cpu')
            strict = pretrained.get('strict', True)
            state_dict = _load_checkpoint_with_prefix(prefix, ckpt_path,
                                                      map_location)
            self.load_state_dict(state_dict, strict=strict)
        elif pretrained is None:
            if self.init_type.upper() == 'STUDIO':
                # initialization method from Pytorch-StudioGAN
                #   * weight: orthogonal_init gain=1
                #   * bias  : 0
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        nn.init.orthogonal_(m.weight, gain=1)
                        if hasattr(m, 'bias') and m.bias is not None:
                            m.bias.data.fill_(0.)
            elif self.init_type.upper() == 'BIGGAN':
                # initialization method from BigGAN-pytorch
                #   * weight: xavier_init gain=1
                #   * bias  : default
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        xavier_uniform_(m.weight, gain=1)
            elif self.init_type.upper() == 'SAGAN':
                # initialization method from official tensorflow code
                #   * weight: xavier_init gain=1
                #   * bias  : 0
                for m in self.modules():
                    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                        xavier_init(m, gain=1, distribution='uniform')
            elif self.init_type.upper() in ['SNGAN', 'SNGAN-PROJ', 'GAN-PROJ']:
                # initialization method from the official chainer code
                #   * embedding.weight: xavier_init gain=1
                #   * conv.weight     : xavier_init gain=sqrt(2)
                #   * shortcut.weight : xavier_init gain=1
                #   * bias            : 0
                for n, m in self.named_modules():
                    if isinstance(m, nn.Conv2d):
                        if 'shortcut' in n:
                            xavier_init(m, gain=1, distribution='uniform')
                        else:
                            xavier_init(
                                m, gain=np.sqrt(2), distribution='uniform')
                    if isinstance(m, (nn.Linear, nn.Embedding)):
                        xavier_init(m, gain=1, distribution='uniform')
            else:
                raise NotImplementedError('Unknown initialization method: '
                                          f'\'{self.init_type}\'')
        else:
            raise TypeError("'pretrained' must by a str or None. "
                            f'But receive {type(pretrained)}.')
