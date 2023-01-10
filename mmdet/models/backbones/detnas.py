import torch.nn as nn
from .shuffle_blocks import ConvBNReLU, ShuffleNetV2BlockSearched, blocks_key, FC
import torch
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES


# class ShuffleNetV2DetNAS(nn.Module):
@BACKBONES.register_module()
class ShuffleNetV2DetNAS(BaseModule):
    def __init__(self, stage_repeats, stage_out_channels, out_indices=(0, 1, 2, 3), init_cfg=None):
        super(ShuffleNetV2DetNAS, self).__init__(init_cfg)
        # print('Model size is {}.'.format(model_size))

        # n_class = 1000
        # if '300M' in model_size:
        #     stage_repeats = [4, 4, 8, 4]
        #     stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        # elif '1.3G' in model_size:
        #     stage_repeats = [8, 8, 16, 8]
        #     stage_out_channels = [-1, 48, 96, 240, 480, 960, 1024]
        # else:
        #     raise NotImplementedError
        self.out_indices = out_indices

        self.stage_repeats = stage_repeats
        self.first_conv = ConvBNReLU(in_channel=3, out_channel=stage_out_channels[1], k_size=3, stride=2, padding=1, gaussian_init=True)

        self.features = list()
        self.stage_ends_idx = list()

        in_channels = stage_out_channels[1]
        i_th = 0
        for id_stage in range(1, len(stage_repeats) + 1):
            out_channels = stage_out_channels[id_stage + 1]
            repeats = stage_repeats[id_stage - 1]
            for id_repeat in range(repeats):
                prefix = str(id_stage) + chr(ord('a') + id_repeat)
                stride = 1 if id_repeat > 0 else 2

                _ops = nn.ModuleList()
                for i in range(len(blocks_key)):
                    _ops.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                           stride=stride, base_mid_channels=out_channels // 2, id=i))
                self.features.append(_ops)

                in_channels = out_channels
                i_th += 1
            self.stage_ends_idx.append(i_th-1)

        self.features = nn.Sequential(*self.features)

        # self.last_conv = ConvBNReLU(in_channel=in_channels, out_channel=stage_out_channels[-1], k_size=1, stride=1, padding=0)
        # self.drop_out = nn.Dropout2d(p=0.2)
        # self.global_pool = nn.AvgPool2d(7)
        # self.fc = FC(in_channels=stage_out_channels[-1], out_channels=n_class)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, arch):
        rngs = arch
        assert sum(self.stage_repeats) == len(rngs)

        outputs = []

        x = self.first_conv(x)

        out_indices = [self.stage_ends_idx[i] for i in self.out_indices]
        # print(self.stage_ends_idx)
        # print(out_indices)

        for i, select_op in enumerate(self.features):
            x =  select_op[rngs[i]](x)
            # if i in self.out_indices:
                # outputs.append(x)
            if i in out_indices:
                # print(i)
                outputs.append(x)

        # x = self.last_conv(x)
        # x = self.drop_out(x)
        # x = self.global_pool(x).view(x.size(0), -1)
        # x = self.fc(x)
        return outputs


@BACKBONES.register_module()
class ShuffleNetV2DetNASSubnet(BaseModule):
    def __init__(self, stage_repeats, stage_out_channels, arch, out_indices=(0, 1, 2, 3), init_cfg=None):
        super(ShuffleNetV2DetNASSubnet, self).__init__(init_cfg)
        # print('Model size is {}.'.format(model_size))

        # n_class = 1000
        # if '300M' in model_size:
        #     stage_repeats = [4, 4, 8, 4]
        #     stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        # elif '1.3G' in model_size:
        #     stage_repeats = [8, 8, 16, 8]
        #     stage_out_channels = [-1, 48, 96, 240, 480, 960, 1024]
        # else:
        #     raise NotImplementedError
        self.out_indices = out_indices

        self.stage_repeats = stage_repeats
        self.first_conv = ConvBNReLU(in_channel=3, out_channel=stage_out_channels[1], k_size=3, stride=2, padding=1, gaussian_init=True)

        self.features = list()
        self.stage_ends_idx = list()

        in_channels = stage_out_channels[1]
        i_th = 0
        for id_stage in range(1, len(stage_repeats) + 1):
            out_channels = stage_out_channels[id_stage + 1]
            repeats = stage_repeats[id_stage - 1]
            for id_repeat in range(repeats):
                prefix = str(id_stage) + chr(ord('a') + id_repeat)
                stride = 1 if id_repeat > 0 else 2
                
                _ops = ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                        stride=stride, base_mid_channels=out_channels // 2, id=arch[i_th])
                self.features.append(_ops)

                in_channels = out_channels
                i_th += 1
            self.stage_ends_idx.append(i_th-1)

        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        outputs = []

        x = self.first_conv(x)

        out_indices = [self.stage_ends_idx[i] for i in self.out_indices]

        for i, select_op in enumerate(self.features):
            x =  select_op(x)
            # if i in self.out_indices:
                # outputs.append(x)
            if i in out_indices:
                # print(i)
                outputs.append(x)

        return outputs