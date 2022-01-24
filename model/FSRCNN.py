# -*- coding: utf-8 -*-
'''

'''
# @Time    : 2022/1/23 15:12
# @Author  : LINYANZHEN
# @File    : FSRCNN.py

import math
from torch import nn


def make_model(args):
    return FSRCNN(args)


class FSRCNN(nn.Module):
    def __init__(self, args, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(3, 56, kernel_size=(5, 5), padding=(2, 2)),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=(1, 1)), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=(3, 3), padding=(1, 1)), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=(1, 1)), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, 3, kernel_size=(9, 9), stride=args.scale, padding=(4, 4),
                                            output_padding=args.scale - 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
