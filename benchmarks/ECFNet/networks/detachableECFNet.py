import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        bias=True,
        norm=False,
        relu=True,
        transpose=False,
    ):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())  # nn.ReLU(inplace=True)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(
            *[
                nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
                # nn.LeakyReLU(negative_slope=0.1)
                # nn.ReLU(inplace=True)
                nn.GELU(),
            ]
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDBlock(nn.Module):
    def __init__(self, in_channel, out_channel, nConvLayers=3):
        super(RDBlock, self).__init__()
        G0 = in_channel
        G = in_channel
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, out_channel, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [RDBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [RDBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF1(nn.Module):
    def __init__(self, in_channel, out_channel, ffn_expansion_factor=2, bias=False):
        super(AFF1, self).__init__()
        hidden_features = int(in_channel * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            in_channel, hidden_features * 2, kernel_size=1, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(
            hidden_features * 2, out_channel, kernel_size=1, bias=bias
        )

    def forward(self, *tensors):
        x = torch.cat(tensors, dim=1)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x1 = F.sigmoid(x1) * x2
        x2 = F.sigmoid(x2) * x1

        x = torch.cat([x1, x2], dim=1)
        x = self.project_out(x)

        return x  # self.conv(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, *tensors):
        x = torch.cat(tensors, dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane, in_nc=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_nc, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(
                out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True
            ),
            BasicConv(
                out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True
            ),
            BasicConv(
                out_plane // 2, out_plane - in_nc, kernel_size=1, stride=1, relu=True
            ),
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, in_nc=3):
        super(SAM, self).__init__()
        self.conv1 = BasicConv(
            n_feat, n_feat, kernel_size=kernel_size, stride=1, relu=True
        )
        self.conv2 = BasicConv(
            n_feat, in_nc, kernel_size=kernel_size, stride=1, relu=False
        )
        self.conv3 = BasicConv(
            in_nc, n_feat, kernel_size=kernel_size, stride=1, relu=False
        )

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class DetachableECFNet(nn.Module):
    AFF_CONSTANTS = [1, 3, 7]

    def __init__(self, in_nc=3, out_nc=3, base_channel=None, level=4):
        super(DetachableECFNet, self).__init__()
        if level not in [1, 2, 3]:
            raise NotImplementedError("Use original ECFNet")

        base_channel = in_nc * 8 if base_channel is None else base_channel
        num_res = 2 * in_nc
        self.level = level

        input_feat_extracts = [
            BasicConv(in_nc, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(
                base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2
            ),
            BasicConv(
                base_channel * 2,
                base_channel * 4,
                kernel_size=3,
                relu=True,
                stride=2,
            ),
        ]
        self.input_feat_extract = nn.ModuleList(input_feat_extracts[:level])

        output_feat_extracts = [
            BasicConv(base_channel, out_nc, kernel_size=3, relu=False, stride=1),
            BasicConv(
                base_channel * 2,
                base_channel,
                kernel_size=4,
                relu=True,
                stride=2,
                transpose=True,
            ),
            BasicConv(
                base_channel * 4,
                base_channel * 2,
                kernel_size=4,
                relu=True,
                stride=2,
                transpose=True,
            ),
        ]
        self.output_feat_extract = nn.ModuleList(output_feat_extracts[:level])

        FAMs = [
            None,
            FAM(base_channel * 2),
            FAM(base_channel * 4),
        ]
        self.FAMs = nn.ModuleList(FAMs[:level])

        encoders = [
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ]
        self.Encoder = nn.ModuleList(encoders[:level])

        decoders = [
            DBlock(base_channel, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel * 4, num_res),
        ]
        self.Decoder = nn.ModuleList(decoders[:level])

        convs = [
            None,
            BasicConv(
                base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1
            ),
            BasicConv(
                base_channel * 2 * 2,
                base_channel * 2,
                kernel_size=1,
                relu=True,
                stride=1,
            ),
        ]
        self.Convs = nn.ModuleList(convs[:level])  # accessing out is okay

        convs_out = [
            None,  # 1
            SAM(base_channel * 2, in_nc=in_nc),  # 2
            SAM(base_channel * 4, in_nc=in_nc),  # 3
        ]
        self.ConvsOut = nn.ModuleList(convs_out[:level])

        aff_constant = self.AFF_CONSTANTS[level - 1]
        affs = [
            AFF(base_channel * aff_constant, base_channel * 1),
            AFF1(base_channel * aff_constant, base_channel * 2),
            AFF1(base_channel * aff_constant, base_channel * 4),
        ]
        self.AFFs = nn.ModuleList(affs[:level])

        scms = [
            None,
            SCM(base_channel * 2, in_nc=in_nc),
            SCM(base_channel * 4, in_nc=in_nc),
        ]
        self.SCMs = nn.ModuleList(scms[:level])

        forward_functions = [
            self._forward_1,
            self._forward_2,
            self._forward_3,
        ]
        self._forward = forward_functions[level - 1]

    def _forward_1(self, x):
        # level 1
        x_ = self.input_feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        # cross gating fusion module
        z = self.AFFs[0](res1)

        # level 1
        z = self.Decoder[0](z)
        z = self.output_feat_extract[0](z)
        o1 = z + x

        return [o1]

    def _forward_2(self, x):
        x2 = F.interpolate(x, scale_factor=0.5)
        z2 = self.SCMs[1](x2)

        # level 1
        x_ = self.input_feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        # level 2
        z = self.input_feat_extract[1](res1)
        z = self.FAMs[1](z, z2)
        res2 = self.Encoder[1](z)

        # CGFM (in level -> out level)
        z11 = res1
        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z22 = res2

        res1 = self.AFFs[0](z11, z21)
        res2 = self.AFFs[1](z12, z22)

        # level 2
        z = self.Decoder[1](res2)
        z, o2 = self.ConvsOut[1](z, x2)
        z = self.output_feat_extract[1](z)

        # level 1
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)  # comes from level 2. thus index is 1
        z = self.Decoder[0](z)
        o1 = self.output_feat_extract[0](z) + x

        return [o1, o2]

    def _forward_3(self, x):
        x2 = F.interpolate(x, scale_factor=0.5)
        x3 = F.interpolate(x2, scale_factor=0.5)

        z2 = self.SCMs[1](x2)
        z3 = self.SCMs[2](x3)

        # level 1
        x_ = self.input_feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        # level 2
        z = self.input_feat_extract[1](res1)
        z = self.FAMs[1](z, z2)
        res2 = self.Encoder[1](z)

        # level 3
        z = self.input_feat_extract[2](res2)
        z = self.FAMs[2](z, z3)
        res3 = self.Encoder[2](z)

        # CGFM
        z11 = res1
        z12 = F.interpolate(res1, scale_factor=0.5)
        z13 = F.interpolate(res1, scale_factor=0.25)

        z21 = F.interpolate(res2, scale_factor=2)
        z22 = res2
        z23 = F.interpolate(res2, scale_factor=0.5)

        z33 = res3
        z32 = F.interpolate(res3, scale_factor=2)
        z31 = F.interpolate(z32, scale_factor=2)

        res1 = self.AFFs[0](z11, z21, z31)
        res2 = self.AFFs[1](z12, z22, z32)
        res3 = self.AFFs[2](z13, z23, z33)

        # level 3
        z = self.Decoder[2](res3)
        z, o3 = self.ConvsOut[2](z, x3)
        z = self.output_feat_extract[2](z)

        # level 2
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[2](z)
        z = self.Decoder[1](z)
        z, o2 = self.ConvsOut[1](z, x2)
        z = self.output_feat_extract[1](z)

        # level 1
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[0](z)
        o1 = self.output_feat_extract[0](z) + x

        return [o1, o2, o3]

    def forward(self, x):
        return self._forward(x)


if __name__ == "__main__":
    patch_size = 256
    channel = 4

    model = ECFNet(in_nc=channel, out_nc=channel).to("cuda:0")
    print(torch.cuda.memory_summary())
    time.sleep(5)

    print("-" * 50)
    print("#generator parameters:", sum(param.numel() for param in model.parameters()))
