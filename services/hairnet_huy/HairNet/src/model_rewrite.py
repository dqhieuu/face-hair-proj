import torch
import torch.nn as nn
import torch.nn.functional as F


class DucModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 8, 2, 3),  # 16 128 128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, 6, 2, 2),  # 64 64 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, 8, 4, 3),  # 64 64 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 6, 2, 2),  # 128 32 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256 16 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 256, 6, 4, 2),  # 256 16 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),  # 512 8 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1),  # 1024 4 4
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 1024, 4, 4, 1),  # 1024 4 4
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.brx = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 100, 1, 1, 0),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True)
        )
        self.bry = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 100, 1, 1, 0),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True)
        )
        self.brz = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 100, 1, 1, 0),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True)
        )
        self.brc = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 100, 1, 1, 0),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True)
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(4, 4, 3, 1, 1),
            nn.BatchNorm3d(4),
            nn.ReLU(inplace=True),
            nn.Conv3d(4, 4, 3, 1, 1),
            nn.BatchNorm3d(4),
            #nn.Tanh()
        )

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(x)
        h12 = h1 + h2
        h3 = self.conv3(h12)
        h4 = self.conv4(h12)
        h34 = h3 + h4
        h5 = self.conv5(h34)
        h6 = self.conv6(h34)
        h0 = h5 + h6
        h0 = F.max_pool2d(h0, 4)
        h0 = h0.view(-1, 256, 2, 2)
        h0 = F.interpolate(h0, scale_factor=2, mode='bilinear', align_corners=False)  # 256 4 4
        h0 = self.conv7(h0)  # 512 4 4
        h0 = F.interpolate(h0, scale_factor=2, mode='bilinear', align_corners=False)  # 512 8 8
        h0 = self.conv8(h0)  # 512 8 8
        h0 = F.interpolate(h0, scale_factor=2, mode='bilinear', align_corners=False)  # 512 16 16
        h0 = self.conv9(h0)  # 512 16 16
        h0 = F.interpolate(h0, scale_factor=2, mode='bilinear', align_corners=False)  # 512 32 32
        h0 = self.conv10(h0)  # 512 32 32

        brx = self.brx(h0)
        brx = brx.view(-1, 1, 100, 32, 32)
        bry = self.bry(h0)
        bry = bry.view(-1, 1, 100, 32, 32)
        brz = self.brz(h0)
        brz = brz.view(-1, 1, 100, 32, 32)
        brc = self.brc(h0)
        brc = brc.view(-1, 1, 100, 32, 32)

        x = [brx, bry, brz, brc]
        x = torch.tanh(torch.cat(x, 1))

        x = self.conv3d(x)
        x = x.permute(0, 2, 1, 3, 4)

        return x  # (batch_size, 100, 4, 32, 32)


class DucModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 8, 2, 3),  # 16 128 128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 64, 6, 2, 2),  # 64 64 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, 8, 4, 3),  # 64 64 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 6, 2, 2),  # 128 32 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256 16 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 256, 6, 4, 2),  # 256 16 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),  # 512 8 8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1),  # 1024 4 4
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 1024, 4, 4, 1),  # 1024 4 4
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.brx = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 300, 1, 1, 0),
            nn.BatchNorm2d(300),
            nn.ReLU(inplace=True)
        )
        self.bry = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 100, 1, 1, 0),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True)
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(4, 4, 3, 1, 1),
            nn.BatchNorm3d(4),
            nn.Tanh(),
            nn.Conv3d(4, 4, 3, 1, 1),
            nn.BatchNorm3d(4),
        )

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(x)
        h12 = h1 + h2
        h3 = self.conv3(h12)
        h4 = self.conv4(h12)
        h34 = h3 + h4
        h5 = self.conv5(h34)
        h6 = self.conv6(h34)
        h0 = h5 + h6
        h0 = F.max_pool2d(h0, 4)
        h0 = h0.view(-1, 256, 2, 2)
        h0 = F.interpolate(h0, scale_factor=2, mode='bilinear', align_corners=False)  # 256 4 4
        h0 = self.conv7(h0)  # 512 4 4
        h0 = F.interpolate(h0, scale_factor=2, mode='bilinear', align_corners=False)  # 512 8 8
        h0 = self.conv8(h0)  # 512 8 8
        h0 = F.interpolate(h0, scale_factor=2, mode='bilinear', align_corners=False)  # 512 16 16
        h0 = self.conv9(h0)  # 512 16 16
        h0 = F.interpolate(h0, scale_factor=2, mode='bilinear', align_corners=False)  # 512 32 32
        h0 = self.conv10(h0)  # 512 32 32

        brx = self.brx(h0)
        brx = brx.view(-1, 3, 100, 32, 32)
        bry = self.bry(h0)
        bry = bry.view(-1, 1, 100, 32, 32)

        x = [brx, bry]
        x = torch.cat(x, 1)

        x = self.conv3d(x)
        x = x.permute(0, 2, 1, 3, 4)

        return x  # (batch_size, 100, 4, 32, 32)


class HairNetModelOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, 2, 3),  # 32 feature maps, 128x128 output
            nn.ReLU(),
            nn.Conv2d(32, 64, 8, 2, 3),  # 64 feature maps, 64x64 output
            nn.ReLU(),
            nn.Conv2d(64, 128, 6, 2, 2),  # 128 feature maps, 32x32 output
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256 feature maps, 16x16 output
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # 256 feature maps, 8x8 output
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),  # 512 feature maps, 8x8 output
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),  # 512 feature maps, 8x8 output
            nn.ReLU(),
            nn.MaxPool2d(8),  # 1x1 output
            nn.Tanh(),
            nn.Flatten(),  # 512 outputs from feature maps
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),  # 256 feature maps, 4x4 output
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
        )

        self.positionDecoder = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),  # 512 feature maps, 32x32 output
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, 1, 0),  # 512 feature maps, 32x32 output
            nn.Tanh(),
            nn.Conv2d(512, 300, 1, 1, 0),  # 512 feature maps, 32x32 output
            nn.Flatten(),  # batchx300x32x32
            nn.Unflatten(1, (100, 3, 32, 32)),  # 32x32 strands, 100 positions(x,y,z) per strand
        )

        self.curvatureDecoder = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),  # 512 feature maps, 32x32 output
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, 1, 0),  # 512 feature maps, 32x32 output
            nn.Tanh(),
            nn.Conv2d(512, 100, 1, 1, 0),  # 512 feature maps, 32x32 output
            nn.Flatten(),  # batchx100x32x32
            nn.Unflatten(1, (100, 1, 32, 32)),
            # 32x32 strands, 100 curvatures(x,y,z) per strand that correspond to the positions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.Sequential(self.encoder, self.decoder)(x)
        positions = self.positionDecoder(x)
        curvatures = self.curvatureDecoder(x)
        return torch.cat([positions, curvatures], 2)


class HairNetLossRewrite(nn.Module):
    def forward(self, output: torch.Tensor, convdata: torch.Tensor, visweight: torch.Tensor):
        pos_cur_loss = PosCurMSERewrite().forward(output, convdata, visweight)
        # col_loss = CollisionLossRewrite().forward(output, convdata)

        return pos_cur_loss


class PosCurMSERewrite(nn.Module):
    # output (compared to) convdata=batchx100x4x32x32
    # visweight batchx100x32x32
    def forward(self, output: torch.Tensor, convdata: torch.Tensor, visweight: torch.Tensor):
        visweight: torch.Tensor = visweight[:, :, :, :].reshape(1, -1)
        visweight = torch.where(visweight > 0, 10.0, 0.1)

        pos_curve_diff_squared: torch.Tensor = torch.pow(convdata - output, 2).reshape(-1, 4)
        pos_curve_sum_loss = visweight.mm(pos_curve_diff_squared).sum()

        return pos_curve_sum_loss / 1024.0


class PosMSERewrite(nn.Module):
    # output (compared to) convdata=batchx100x4x32x32
    # visweight batchx100x32x32
    def forward(self, output: torch.Tensor, convdata: torch.Tensor, visweight: torch.Tensor):
        visweight: torch.Tensor = visweight[:, :, :, :].reshape(1, -1)
        visweight = torch.where(visweight > 0, 10.0, 0.1)

        e_squared: torch.Tensor = torch.pow(
            (convdata[:, :, 0:3, :, :] - output[:, :, 0:3, :, :]), 2
        ).reshape(-1, 3)
        loss = visweight.mm(e_squared).sum()

        return loss / 1024.0


class CurMSERewrite(nn.Module):
    # output (compared to) convdata=batchx100x4x32x32
    # visweight batchx100x32x32
    def forward(self, output, convdata, visweight):
        visweight = visweight[:, :, :, :].reshape(1, -1)
        visweight = torch.where(visweight > 0, 10.0, 0.1)
        e_squared = torch.pow(
            (convdata[:, :, 3, :, :] - output[:, :, 3, :, :]), 2
        ).reshape(-1, 1)
        loss = visweight.mm(e_squared).sum()

        return loss / 1024.0

# class CollisionLossRewrite(nn.Module):
#     # output (compared to) convdata=batchx100x4x32x32
#     def forward(self, output, convdata):
#         x, y, z = 0.005, 1.75, 0.01
#         a, b, c = 0.08, 0.12, 0.1
#         L1 = torch.add(
#             torch.add(
#                 torch.abs(output[:, :, 0, :, 1:] - output[:, :, 0, :, :-1]),
#                 torch.abs(output[:, :, 1, :, 1:] - output[:, :, 1, :, :-1]),
#             ),
#             torch.abs(output[:, :, 2, :, 1:] - output[:, :, 2, :, :-1]),
#         )
#         D = 1 - torch.add(
#             torch.add(
#                 torch.pow((output[:, :, 0, :, 1:] - x) / a, 2),
#                 torch.pow((output[:, :, 1, :, 1:] - y) / b, 2),
#             ),
#             torch.pow((output[:, :, 2, :, 1:] - z) / c, 2),
#         )
#         D[D < 0] = 0
#         C = torch.sum(L1 * D)
#         return C / 1024.0
#         # loss = C / (convdata.shape[0] * convdata.shape[1] * 1024.0)
#         # return loss

# class MyLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, output: torch.Tensor, convdata: torch.Tensor, visweight: torch.Tensor):
#         # removing nested for-loops (0.238449s -> 0.001860s)
#         pos_loss = PosMSE().forward(output, convdata, visweight)
#         cur_loss = CurMSE().forward(output, convdata, visweight)
#         col_loss = CollisionLoss().forward(output, convdata)
#
#         return pos_loss + cur_loss + col_loss
#
#
# class CollisionLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, output, convdata):
#         x, y, z = 0.005, 1.75, 0.01
#         a, b, c = 0.08, 0.12, 0.1
#         L1 = torch.add(
#             torch.add(
#                 torch.abs(output[:, :, 0, :, 1:] - output[:, :, 0, :, :-1]),
#                 torch.abs(output[:, :, 1, :, 1:] - output[:, :, 1, :, :-1]),
#             ),
#             torch.abs(output[:, :, 2, :, 1:] - output[:, :, 2, :, :-1]),
#         )
#         D = 1 - torch.add(
#             torch.add(
#                 torch.pow((output[:, :, 0, :, 1:] - x) / a, 2),
#                 torch.pow((output[:, :, 1, :, 1:] - y) / b, 2),
#             ),
#             torch.pow((output[:, :, 2, :, 1:] - z) / c, 2),
#         )
#         D[D < 0] = 0
#         C = torch.sum(L1 * D)
#         return C / 1024.0
#         # loss = C / (convdata.shape[0] * convdata.shape[1] * 1024.0)
#         # return loss
#
#
# class PosMSE(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, output: torch.Tensor, convdata: torch.Tensor, visweight: torch.Tensor):
#         visweight = visweight[:, :, :, :].reshape(1, -1)
#         e_squared: torch.Tensor = torch.pow(
#             (convdata[:, :, 0:3, :, :] - output[:, :, 0:3, :, :]), 2
#         ).reshape(-1, 3)
#         loss = visweight.mm(e_squared).sum()
#
#         # convdata.shape 8, 100, 4, 32, 32
#         return loss / 1024
#
#         # return loss / (convdata.shape[0] * convdata.shape[1] * 1024.0)
#
#
# class CurMSE(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, output, convdata, visweight):
#         visweight = visweight[:, :, :, :].reshape(1, -1)
#         e_squared = torch.pow(
#             (convdata[:, :, 3, :, :] - output[:, :, 3, :, :]), 2
#         ).reshape(-1, 1)
#         loss = visweight.mm(e_squared).sum()
#
#         return loss / 1024
#
#         # return loss / (convdata.shape[0] * convdata.shape[1] * 1024.0)

# def run():
#     # load data
#     conv1 = np.load("../convdata/strands00009_00026_00000.convdata")
#     conv2 = np.load("../convdata/strands00009_00026_00000.convdata")
#     visweight = np.load("../data/strands00009_00026_00000_v0.vismap")
#     res = MyLoss().forward(torch.from_numpy(conv1).view(1,100,4,32,32), torch.from_numpy(conv2).view(1,100,4,32,32),torch.from_numpy(visweight).view(1,100, 32, 32))
#     print(res)
#
# if __name__ == "__main__":
#     run()


#
# class HairNetModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, 8, 2, 3),  # 32 feature maps, 128x128 output
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 4, 2, 1),  # 64 feature maps, 64x64 output
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 4, 2, 1),  # 128 feature maps, 32x32 output
#             nn.ReLU(),
#             nn.Conv2d(128, 256, 2, 2, 0),  # 256 feature maps, 16x16 output
#             nn.ReLU(),
#             nn.Conv2d(256, 512, 2, 2, 0),  # 512 feature maps, 8x8 output
#             nn.ReLU(),
#             nn.MaxPool2d(8, 1, 0),  # 1x1 output
#             nn.Flatten(),  # 512 outputs from feature maps
#         )
#
#         self.decoder = nn.Sequential(
#             nn.Linear(512, 4096),  # expand the feature maps 8 times
#             nn.Unflatten(1, (256, 4, 4)),  # 256 feature maps, 4x4 output
#             nn.ConvTranspose2d(256, 512, 4, 2, 1),  # 512 feature maps, 8x8 output
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 512 feature maps, 16x16 output
#             nn.ReLU(),
#             nn.ConvTranspose2d(512, 512, 6, 2, 2),  # 512 feature maps, 32x32 output
#             nn.ReLU(),
#         )
#
#         self.positionMLP = nn.Sequential(
#             nn.Conv2d(512, 512, 1, 1, 0),  # 512 feature maps, 32x32 output
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 1, 1, 0),  # 512 feature maps, 32x32 output
#             nn.ReLU(),
#             nn.Conv2d(512, 300, 1, 1, 0),  # 512 feature maps, 32x32 output
#             nn.ReLU(),
#             nn.Flatten(), # batchx300x32x32
#             nn.Unflatten(1, (100, 3, 32, 32)),  # 32x32 strands, 100 positions(x,y,z) per strand
#         )
#
#         self.curvatureMLP = nn.Sequential(
#             nn.Conv2d(512, 512, 1, 1, 0),  # 512 feature maps, 32x32 output
#             nn.ReLU(),
#             nn.Conv2d(512, 512, 1, 1, 0),  # 512 feature maps, 32x32 output
#             nn.ReLU(),
#             nn.Conv2d(512, 100, 1, 1, 0),  # 512 feature maps, 32x32 output
#             nn.ReLU(),
#             nn.Flatten(),  # batchx100x32x32
#             nn.Unflatten(1, (100, 1, 32, 32)),  # 32x32 strands, 100 curvatures(x,y,z) per strand that correspond to the positions
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = nn.Sequential(self.encoder, self.decoder)(x)
#         positions = self.positionMLP(x)
#         curvatures = self.curvatureMLP(x)
#         return torch.cat([positions, curvatures], 2)

# huy legacy code
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # encoder
#         # self.conv1 = nn.Conv2d(3, 32, 8, 2, 3)
#         self.conv2 = nn.Conv2d(3, 64, 8, 2, 3)
#         self.conv3 = nn.Conv2d(64, 128, 6, 2, 2)
#         self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
#         self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)
#         # decoder
#         self.fc1 = nn.Linear(512, 4096)
#         # self.fc2 = nn.Linear(1024, 4096)
#         self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
#         self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
#         self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)
#         # MLP
#         # Position
#         self.branch1_fc1 = nn.Conv2d(512, 512, 1, 1, 0)
#         self.branch1_fc2 = nn.Conv2d(512, 512, 1, 1, 0)
#         self.branch1_fc3 = nn.Conv2d(512, 300, 1, 1, 0)
#         # Curvature
#         self.branch2_fc1 = nn.Conv2d(512, 512, 1, 1, 0)
#         self.branch2_fc2 = nn.Conv2d(512, 512, 1, 1, 0)
#         self.branch2_fc3 = nn.Conv2d(512, 100, 1, 1, 0)
#
#     def forward(self, x, interp_factor=1):
#         # encoder
#         # x = F.relu(self.conv1(x)) # (batch_size, 32, 128, 128)
#         x = F.relu(self.conv2(x))  # (batch_size, 64, 64, 64)
#         x = F.relu(self.conv3(x))  # (batch_size, 128, 32, 32)
#         x = F.relu(self.conv4(x))  # (batch_size, 256, 16, 16)
#         x = F.relu(self.conv5(x))  # (batch_size, 512, 8, 8)
#         x = torch.tanh(F.max_pool2d(x, 8))  # (batch_size, 512, 1, 1)
#         # decoder
#         x = x.view(-1, 1 * 1 * 512)
#         x = F.relu(self.fc1(x))
#         # x = x.view(-1, 1*1*1024)
#         # x = F.relu(self.fc2(x))
#         x = x.view(-1, 256, 4, 4)
#         x = F.relu(self.conv6(x))  # (batch_size, 256, 4, 4)
#         x = F.interpolate(
#             x, scale_factor=2, mode="bilinear", align_corners=False
#         )  # (batch_size, 256, 8, 8)
#         x = F.relu(self.conv7(x))  # (batch_size, 256, 8, 8)
#         x = F.interpolate(
#             x, scale_factor=2, mode="bilinear", align_corners=False
#         )  # (batch_size, 256, 16, 16)
#         x = F.relu(self.conv8(x))  # (batch_size, 256, 16, 16)
#         x = F.interpolate(
#             x, scale_factor=2, mode="bilinear", align_corners=False
#         )  # (batch_size, 256, 32, 32)
#         # interpolate feature map
#         if interp_factor != 1:
#             x = F.interpolate(
#                 x, scale_factor=interp_factor, mode="bilinear", align_corners=False
#             )  # (batch_size, 256, 32, 32)
#         # MLP
#         # Position
#         branch1_x = F.relu(self.branch1_fc1(x))
#         branch1_x = F.relu(self.branch1_fc2(branch1_x))
#         branch1_x = self.branch1_fc3(branch1_x)
#         branch1_x = branch1_x.view(-1, 100, 3, 32 * interp_factor, 32 * interp_factor)
#         # Curvature
#         branch2_x = F.relu(self.branch2_fc1(x))
#         branch2_x = F.relu(self.branch2_fc2(branch2_x))
#         branch2_x = self.branch2_fc3(branch2_x)
#         branch2_x = branch2_x.view(-1, 100, 1, 32 * interp_factor, 32 * interp_factor)
#         x = torch.cat([branch1_x, branch2_x], 2)
#         return x  # (batch_size, 100, 4, 32, 32)
