"""
Small HRNet for ball detection (WASB architecture).

Reference: "WASB-SBDT: Joint Use of Weak Annotations and Small Ball
Detector for Tracking" (arXiv:2311.05237).

Key difference from standard HRNet: stride-free stem (both conv strides = 1)
so the model operates at full input resolution.

Attribute names match the original checkpoint exactly so that weights load
without any key remapping.
"""

import torch
import torch.nn as nn

BN_MOMENTUM = 0.1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


BLOCKS = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels, multi_scale_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.num_inchannels = list(num_inchannels)
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_idx, block, num_blocks, num_channels):
        downsample = None
        if self.num_inchannels[branch_idx] != num_channels[branch_idx] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_idx], num_channels[branch_idx] * block.expansion, 1, bias=False),
                nn.BatchNorm2d(num_channels[branch_idx] * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_idx], num_channels[branch_idx], 1, downsample)]
        self.num_inchannels[branch_idx] = num_channels[branch_idx] * block.expansion
        for _ in range(1, num_blocks[branch_idx]):
            layers.append(block(self.num_inchannels[branch_idx], num_channels[branch_idx]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        fuse_layers = []
        out_branches = self.num_branches if self.multi_scale_output else 1
        for i in range(out_branches):
            layer = []
            for j in range(self.num_branches):
                if j > i:
                    layer.append(
                        nn.Sequential(
                            nn.Conv2d(self.num_inchannels[j], self.num_inchannels[i], 1, bias=False),
                            nn.BatchNorm2d(self.num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    layer.append(None)
                else:
                    convs = []
                    for k in range(i - j):
                        out_ch = self.num_inchannels[i] if k == i - j - 1 else self.num_inchannels[j]
                        modules = [
                            nn.Conv2d(self.num_inchannels[j], out_ch, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_ch),
                        ]
                        if k < i - j - 1:
                            modules.append(nn.ReLU(True))
                        convs.append(nn.Sequential(*modules))
                    layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class WASBHRNet(nn.Module):
    """
    Small HRNet with stride-free stem for ball heatmap prediction.

    Input:  (B, 9, 288, 512) — 3 consecutive RGB frames concatenated along C.
    Output: (B, 3, 288, 512) — one heatmap (logits, pre-sigmoid) per input frame.

    Attribute names (transition1/2/3, stage2/3/4, layer1, final_layers, etc.)
    match the original HRNet checkpoint format exactly.
    """

    FRAMES_IN = 3
    FRAMES_OUT = 3
    STEM_INPLANES = 64
    STEM_STRIDES = (1, 1)

    STAGE_CFGS = {
        1: {"block": "BOTTLENECK", "num_modules": 1, "num_branches": 1, "num_blocks": [1], "num_channels": [32]},
        2: {"block": "BASIC", "num_modules": 1, "num_branches": 2, "num_blocks": [2, 2], "num_channels": [16, 32]},
        3: {
            "block": "BASIC",
            "num_modules": 1,
            "num_branches": 3,
            "num_blocks": [2, 2, 2],
            "num_channels": [16, 32, 64],
        },
        4: {
            "block": "BASIC",
            "num_modules": 1,
            "num_branches": 4,
            "num_blocks": [2, 2, 2, 2],
            "num_channels": [16, 32, 64, 128],
        },
    }

    def __init__(self):
        super().__init__()

        # ---- Stride-free stem ------------------------------------------------
        self.conv1 = nn.Conv2d(
            3 * self.FRAMES_IN, self.STEM_INPLANES, 3, stride=self.STEM_STRIDES[0], padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.STEM_INPLANES, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            self.STEM_INPLANES, self.STEM_INPLANES, 3, stride=self.STEM_STRIDES[1], padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.STEM_INPLANES, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # ---- Stage 1 --------------------------------------------------------
        s1 = self.STAGE_CFGS[1]
        block1 = BLOCKS[s1["block"]]
        self.layer1 = self._make_layer(block1, self.STEM_INPLANES, s1["num_channels"][0], s1["num_blocks"][0])
        stage1_out = block1.expansion * s1["num_channels"][0]

        # ---- Stage 2 --------------------------------------------------------
        s2 = self.STAGE_CFGS[2]
        block2 = BLOCKS[s2["block"]]
        s2_channels = [c * block2.expansion for c in s2["num_channels"]]
        self.transition1 = self._make_transition([stage1_out], s2_channels)
        self.stage2, pre_ch = self._make_stage(s2, s2_channels)

        # ---- Stage 3 --------------------------------------------------------
        s3 = self.STAGE_CFGS[3]
        block3 = BLOCKS[s3["block"]]
        s3_channels = [c * block3.expansion for c in s3["num_channels"]]
        self.transition2 = self._make_transition(pre_ch, s3_channels)
        self.stage3, pre_ch = self._make_stage(s3, s3_channels)

        # ---- Stage 4 --------------------------------------------------------
        s4 = self.STAGE_CFGS[4]
        block4 = BLOCKS[s4["block"]]
        s4_channels = [c * block4.expansion for c in s4["num_channels"]]
        self.transition3 = self._make_transition(pre_ch, s4_channels)
        self.stage4, pre_ch = self._make_stage(s4, s4_channels, multi_scale_output=True)

        # ---- Head (1x1 conv, scale 0 only) ----------------------------------
        self.final_layers = nn.ModuleList(
            [
                nn.Conv2d(pre_ch[0], self.FRAMES_OUT, 1),
            ]
        )

        self._init_weights()

    # ---- building helpers ---------------------------------------------------

    @staticmethod
    def _make_layer(block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_transition(pre_channels, cur_channels):
        n_pre = len(pre_channels)
        n_cur = len(cur_channels)
        layers = []
        for i in range(n_cur):
            if i < n_pre:
                if cur_channels[i] != pre_channels[i]:
                    layers.append(
                        nn.Sequential(
                            nn.Conv2d(pre_channels[i], cur_channels[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(cur_channels[i], momentum=BN_MOMENTUM),
                            nn.ReLU(True),
                        )
                    )
                else:
                    layers.append(None)
            else:
                convs = []
                for j in range(i + 1 - n_pre):
                    in_ch = pre_channels[-1]
                    out_ch = cur_channels[i] if j == i - n_pre else in_ch
                    convs.append(
                        nn.Sequential(
                            nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM),
                            nn.ReLU(True),
                        )
                    )
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers)

    @staticmethod
    def _make_stage(scfg, num_inchannels, multi_scale_output=True):
        block = BLOCKS[scfg["block"]]
        modules = []
        ch = list(num_inchannels)
        for i in range(scfg["num_modules"]):
            ms = multi_scale_output if i == scfg["num_modules"] - 1 else True
            mod = HighResolutionModule(
                scfg["num_branches"],
                block,
                scfg["num_blocks"],
                ch,
                scfg["num_channels"],
                ms,
            )
            ch = mod.num_inchannels
            modules.append(mod)
        return nn.Sequential(*modules), ch

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ---- forward ------------------------------------------------------------

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        # stage 2
        x_list = []
        for i in range(self.STAGE_CFGS[2]["num_branches"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # stage 3
        x_list = []
        for i in range(self.STAGE_CFGS[3]["num_branches"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # stage 4
        x_list = []
        for i in range(self.STAGE_CFGS[4]["num_branches"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return self.final_layers[0](y_list[0])
