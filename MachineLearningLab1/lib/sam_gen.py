#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
    生成实验用的样本数据，并存储到相同目录下的test_sample.txt和validate_sample.txt文件中。
    生成样本数据使用的函数为y = sin(2pix),加上分布为N(0,1)的噪声。
'''
import numpy as np
import math
from time import time


class SamGen(object):
    def __init__(self):
        self.file = None
        self.num = 0

    # 为生成器设置参数
    def set_para(self, file, num):
        self.file = file
        self.num = num

    # 生成器，用于生成样本数据
    def _gen_sin2pix(self, size):
        np.random.seed(int(time()))
        noi = np.random.normal(0, 0.1, size)
        ran_x = np.random.uniform(-1, 1, size)
        for i in range(size):
            yield ran_x[i], math.sin(2 * math.pi * (ran_x[i])) + noi[i]
        return

    # 将生成的数据写入文件中
    def gen_sample(self):
        file = open(self.file, 'w')
        if not file:
            print('[-]create test sample file failed.')
            exit(1)
        for x, y in self._gen_sin2pix(self.num):
            file.write(str(x) + '\t' + str(y) + '\n')
        file.close()
        return


# 测试代码，用于观察程序的正确性
if __name__ == '__main__':
    s = SamGen()
    s.gen()
