import torch
from torch import nn


class SKConv(nn.Module):
    def __init__(self, features, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch,dropout=False):
        super(up_conv, self).__init__()
        self.upsample=nn.Upsample(scale_factor=2)
        self.conv=nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout2d(p=0.5, inplace=False)
        self.dp=dropout
        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x=self.upsample(x)
        if self.dp:
            x=self.dropout(x)
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return x

class SK_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SK_conv_block, self).__init__()
        mid_features = int(out_ch / 2)
        self.conv = nn.Sequential(
            # SKConv(in_ch, mid_features, 2, 8, 2),
            nn.Conv2d(in_ch, mid_features, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_features, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UncertaintyNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=1, numclasses=4,dropout=False):
        super(UncertaintyNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # probability path
        self.Up5_probability = up_conv(filters[4], filters[3],dropout=dropout)
        self.Up_conv5_probability = SK_conv_block(filters[4], filters[3])

        self.Up4_probability = up_conv(filters[3], filters[2],dropout=dropout)
        self.Up_conv4_probability = SK_conv_block(filters[3], filters[2])

        self.Up3_probability = up_conv(filters[2], filters[1],dropout=dropout)
        self.Up_conv3_probability = SK_conv_block(filters[2], filters[1])

        self.Up2_probability = up_conv(filters[1], filters[0],dropout=dropout)
        self.Up_conv2_probability = SK_conv_block(filters[1], filters[0])

        self.Conv_probability = nn.Conv2d(filters[0], numclasses, kernel_size=1, stride=1, padding=0)

        # distance path
        self.Up5_distance = up_conv(filters[4], filters[3],dropout=dropout)
        self.Up_conv5_distance = SK_conv_block(filters[4], filters[3])

        self.Up4_distance = up_conv(filters[3], filters[2],dropout=dropout)
        self.Up_conv4_distance = SK_conv_block(filters[3], filters[2])

        self.Up3_distance = up_conv(filters[2], filters[1],dropout=dropout)
        self.Up_conv3_distance = SK_conv_block(filters[2], filters[1])

        self.Up2_distance = up_conv(filters[1], filters[0],dropout=dropout)
        self.Up_conv2_distance = SK_conv_block(filters[1], filters[0])

        self.Conv_distance = nn.Conv2d(filters[0], numclasses-1, kernel_size=1, stride=1, padding=0)

        # contour path
        self.Up5_contour = up_conv(filters[4], filters[3],dropout=dropout)
        self.Up_conv5_contour = SK_conv_block(filters[4], filters[3])

        self.Up4_contour = up_conv(filters[3], filters[2],dropout=dropout)
        self.Up_conv4_contour = SK_conv_block(filters[3], filters[2])

        self.Up3_contour = up_conv(filters[2], filters[1],dropout=dropout)
        self.Up_conv3_contour = SK_conv_block(filters[2], filters[1])

        self.Up2_contour = up_conv(filters[1], filters[0],dropout=dropout)
        self.Up_conv2_contour = SK_conv_block(filters[1], filters[0])

        self.Conv_contour = nn.Conv2d(filters[0], numclasses-1, kernel_size=1, stride=1, padding=0)
    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # probability path
        d5_probability = self.Up5_probability(e5)
        d5_probability= torch.cat((e4, d5_probability), dim=1)

        d5_probability = self.Up_conv5_probability(d5_probability)

        d4_probability = self.Up4_probability(d5_probability)
        d4_probability = torch.cat((e3, d4_probability), dim=1)
        d4_probability = self.Up_conv4_probability(d4_probability)

        d3_probability = self.Up3_probability(d4_probability)
        d3_probability = torch.cat((e2, d3_probability), dim=1)
        d3_probability = self.Up_conv3_probability(d3_probability)

        d2_probability = self.Up2_probability(d3_probability)
        d2_probability = torch.cat((e1, d2_probability), dim=1)
        d2_probability = self.Up_conv2_probability(d2_probability)

        out_probability = self.Conv_probability(d2_probability)

        # distance path
        d5_distance = self.Up5_distance(e5)
        d5_distance = torch.cat((e4, d5_distance), dim=1)

        d5_distance = self.Up_conv5_distance(d5_distance)

        d4_distance = self.Up4_distance(d5_distance)
        d4_distance = torch.cat((e3, d4_distance), dim=1)
        d4_distance = self.Up_conv4_distance(d4_distance)

        d3_distance = self.Up3_distance(d4_distance)
        d3_distance = torch.cat((e2, d3_distance), dim=1)
        d3_distance = self.Up_conv3_distance(d3_distance)

        d2_distance = self.Up2_distance(d3_distance)
        d2_distance = torch.cat((e1, d2_distance), dim=1)
        d2_distance = self.Up_conv2_distance(d2_distance)

        out_distance = self.Conv_distance(d2_distance)

        # contour path
        d5_contour = self.Up5_contour(e5)
        d5_contour = torch.cat((e4, d5_contour), dim=1)

        d5_contour = self.Up_conv5_contour(d5_contour)

        d4_contour = self.Up4_contour(d5_contour)
        d4_contour = torch.cat((e3, d4_contour), dim=1)
        d4_contour = self.Up_conv4_contour(d4_contour)

        d3_contour = self.Up3_contour(d4_contour)
        d3_contour = torch.cat((e2, d3_contour), dim=1)
        d3_contour = self.Up_conv3_contour(d3_contour)

        d2_contour = self.Up2_contour(d3_contour)
        d2_contour = torch.cat((e1, d2_contour), dim=1)
        d2_contour = self.Up_conv2_contour(d2_contour)

        out_contour = self.Conv_contour(d2_contour)

        return out_probability,out_distance,out_contour


