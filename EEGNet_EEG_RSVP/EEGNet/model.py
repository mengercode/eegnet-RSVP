#
# # -*- coding: utf-8 -*-
# """
# Created on 2020年7月31日
# @author: Tamie Li
# @description: use pytorch to reproduce EEGNet
# """
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    four block:
    1. conv2d
    2. depthwiseconv2d
    3. separableconv2d
    4. classify
    """
    def __init__(self, batch_size=4, num_class=2):
        super(EEGNet, self).__init__()
        self.batch_size = batch_size
        # 1. conv2d
        self.block1 = nn.Sequential()
        self.block1_conv = nn.Conv2d(in_channels=1,
                                     out_channels=8,
                                     kernel_size=(1, 50),
                                     padding=(0, 25),
                                     bias=False
                                     )
        self.block1.add_module('conv1', self.block1_conv)
        self.block1.add_module('norm1', nn.BatchNorm2d(8))

        # 2. depthwiseconv2d
        self.block2 = nn.Sequential()
        # [N, 8, 62, 250] -> [N, 16, 1, 250]
        self.block2.add_module('conv2', nn.Conv2d(in_channels=8,
                                                  out_channels=16,
                                                  kernel_size=(62, 1),
                                                  groups=2,
                                                  bias=False))
        self.block2.add_module('act1', nn.ELU())
        # [N, 16, 1, 250] -> [N, 16, 1, 50]
        self.block2.add_module('pool1', nn.AvgPool2d(kernel_size=(1, 5)))
        self.block2.add_module('drop1', nn.Dropout(p=0.5))

        # 3. separableconv2d
        self.block3 = nn.Sequential()
        self.block3.add_module('conv3', nn.Conv2d(in_channels=16,
                                                  out_channels=16,
                                                  kernel_size=(1, 16),
                                                  padding=(0, 8),
                                                  groups=16,
                                                  bias=False
                                                  ))
        self.block3.add_module('conv4', nn.Conv2d(in_channels=16,
                                                  out_channels=16,
                                                  kernel_size=(1, 1),
                                                  bias=False))
        self.block3.add_module('norm2', nn.BatchNorm2d(16))
        self.block3.add_module('act2', nn.ELU())
        self.block3.add_module('pool2', nn.AvgPool2d(kernel_size=(1, 10)))
        self.block3.add_module('drop2', nn.Dropout(p=0.5))

        # 4. classify
        self.classify = nn.Sequential(nn.Linear(80, num_class))

    def forward(self, x):
        # 动态获取批量大小
        current_batch_size = x.size(0)

        # 调整输入张量的形状
        x = x.view(current_batch_size, 1, 62, 250)  # 使用动态批量大小

        # [B, 1, 62, 250] -> [B, 1, 62, 249]
        # because pytorch's padding does not have the same option,
        # remove one column before convolution
        x = x[:, :, :, range(249)]

        # [B, 1, 62, 250] -> [B, 8, 62, 250]
        x = self.block1(x)

        # [B, 8, 64, 250] -> [B, 16, 1, 250] -> [B, 16, 1, 50]
        x = self.block2(x)

        # [B, 16, 1, 50] -> [B, 16, 1, 49]
        # because pytorch's padding does not have the same option,
        # remove one column before convolution
        x = x[:, :, :, range(49)]

        # [B, 16, 1, 49] -> [B, 16, 1, 5]
        x = self.block3(x)

        # [B, 16, 1, 5] -> [B, 80]
        x = x.view(x.size(0), -1)

        # [B, 80] -> [B, num_class]
        x = self.classify(x)

        # x = nn.functional.softmax(x, dim=1)

        return x

#

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class EEGNet(nn.Module):
#     def __init__(self, num_channels=64, num_samples=250, num_classes=2, D=2):
#         super(EEGNet, self).__init__()
#
#         # 1. conv2d
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 51), padding=(0, 25), bias=False),
#             nn.BatchNorm2d(8)
#         )
#
#         # 2. depthwiseconv2d
#         self.block2 = nn.Sequential(
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(num_channels, 1), groups=8, bias=False),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, 5)),  # Adjusted kernel size
#             nn.Dropout(p=0.5)
#         )
#
#         # 3. separableconv2d
#         self.block3 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16), padding=(0, 8), groups=16, bias=False),
#             nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), bias=False),
#             nn.BatchNorm2d(16),
#             nn.ELU(),
#             nn.AvgPool2d(kernel_size=(1, 10)),  # Adjusted kernel size
#             nn.Dropout(p=0.5)
#         )
#
#         # 4. classify
#         self.classify = nn.Sequential(nn.Linear(16 * (num_samples // 50), num_classes))  # Adjusted linear layer input
#
#     def forward(self, x):
#         # Adjust input shape to [batch_size, 1, 64, 250]
#         x = x.view(-1, 1, 64, 250)  # Removed batch_size from reshape
#
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#
#         # Flatten the output for the linear layer
#         x = x.view(x.size(0), -1)
#
#         x = self.classify(x)
#
#         return F.log_softmax(x, dim=1)
#
# #     def forward(self, x):
# #         x = self.first_conv(x)
# #         x = self.batchnorm1(x)
# #         x = self.depthwise_conv(x)
# #         x = self.batchnorm2(x)
# #         x = self.activation(x)
# #         x = self.avg_pool1(x)
# #         x = self.dropout1(x)
# #         x = self.separable_conv(x)
# #         x = self.batchnorm3(x)
# #         x = self.activation(x)
# #         x = self.avg_pool2(x)
# #         x = self.dropout2(x)
# #         x = self.flatten(x)
# #         x = self.dense(x)
# #         return F.log_softmax(x, dim=1)
# #
# #
# #
# # """
# # Created on 2020年7月31日
# # @author: Tamie Li
# # @description: use pytorch to reproduce EEGNet
# # """
# # import torch
# # import torch.nn as nn
# #
# #
# # class EEGNet(nn.Module):
# #     """
# #     four block:
# #     1. conv2d
# #     2. depthwiseconv2d
# #     3. separableconv2d
# #     4. classify
# #     """
# #     def __init__(self, batch_size=4, num_class=2):
# #         super(EEGNet, self).__init__()
# #         self.batch_size = batch_size
# #         # 1. conv2d
# #         self.block1 = nn.Sequential()
# #         self.block1_conv = nn.Conv2d(in_channels=1,
# #                                      out_channels=8,
# #                                      kernel_size=(1, 64),
# #                                      padding=(0, 32),
# #                                      bias=False
# #                                      )
# #         self.block1.add_module('conv1', self.block1_conv)
# #         self.block1.add_module('norm1', nn.BatchNorm2d(8))
# #
# #         # 2. depthwiseconv2d
# #         self.block2 = nn.Sequential()
# #         # [N, 8, 64, 250] -> [N, 16, 1, 250]
# #         self.block2.add_module('conv2', nn.Conv2d(in_channels=8,
# #                                                   out_channels=16,
# #                                                   kernel_size=(64, 1),
# #                                                   groups=2,
# #                                                   bias=False))
# #         self.block2.add_module('act1', nn.ELU())
# #         # [N, 16, 1, 250] -> [N, 16, 1, 50]
# #         self.block2.add_module('pool1', nn.AvgPool2d(kernel_size=(1, 5)))
# #         self.block2.add_module('drop1', nn.Dropout(p=0.5))
# #
# #         # 3. separableconv2d
# #         self.block3 = nn.Sequential()
# #         self.block3.add_module('conv3', nn.Conv2d(in_channels=16,
# #                                                   out_channels=16,
# #                                                   kernel_size=(1, 16),
# #                                                   padding=(0, 8),
# #                                                   groups=16,
# #                                                   bias=False
# #                                                   ))
# #         self.block3.add_module('conv4', nn.Conv2d(in_channels=16,
# #                                                   out_channels=16,
# #                                                   kernel_size=(1, 1),
# #                                                   bias=False))
# #         self.block3.add_module('norm2', nn.BatchNorm2d(16))
# #         self.block3.add_module('act2', nn.ELU())
# #         self.block3.add_module('pool2', nn.AvgPool2d(kernel_size=(1, 10)))
# #         self.block3.add_module('drop2', nn.Dropout(p=0.5))
# #
# #         # 4. classify
# #         self.classify = nn.Sequential(nn.Linear(80, num_class))
# #
# #     def forward(self, x):
# #         # [B, 64, 250] -> [B, 1, 64, 250]
# #         x = torch.reshape(x, (self.batch_size, 1, 64, 250))
# #
# #         # [B, 1, 64, 250] -> [B, 1, 64, 249]
# #         # because pytorch's padding does not have the same option,
# #         # remove one column before convolution
# #         x = x[:, :, :, range(249)]
# #
# #         # [B, 1, 64, 250] -> [B, 8, 64, 250]
# #         x = self.block1(x)
# #
# #         # [B, 8, 64, 250] -> [B, 16, 1, 250] -> [B, 16, 1, 50]
# #         x = self.block2(x)
# #
# #         # [B, 16, 1, 50] -> [B, 16, 1, 49]
# #         # because pytorch's padding does not have the same option,
# #         # remove one column before convolution
# #         x = x[:, :, :, range(49)]
# #
# #         # [B, 16, 1, 49] -> [B, 16, 1, 5]
# #         x = self.block3(x)
# #
# #         # [B, 16, 1, 5] -> [B, 80]
# #         x = x.view(x.size(0), -1)
# #
# #         # [B, 80] -> [B, num_class]
# #         x = self.classify(x)
# #
# #         # x = nn.functional.softmax(x, dim=1)
# #
# #         return x
#
