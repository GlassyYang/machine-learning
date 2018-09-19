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
        # 测试样本的数量
        self.test_sample = 10
        # 验证样本的数量
        self.vali_sample = 30

    # 生成器，用于生成样本数据
    def _gen_sin2pix(self, size):
        np.random.seed(int(time()))
        noi = np.random.randn(size)
        ran_x = np.random.uniform(0,2, size)
        for i in range(size):
            yield ran_x[i], math.sin(2 * math.pi * (ran_x[i] + noi[i]))
        return

    # 将生成的数据写入文件中
    def _gen_sample(self, file, size):
        file = open(file, 'w');
        if not file:
            print('[-]create test sample file failed.')
            exit(1)
        for x, y in self._gen_sin2pix(size):
            file.write(str(x) + '\t' + str(y) + '\n')
        file.close();
        return

    # 调用函数生成样本数据并写入到文件
    def gen(self):
        self._gen_sample('text_sample.txt',self.test_sample)
        self._gen_sample('validate_sample.txt', self.vali_sample)
        return


# 测试代码，用于观察程序的正确性
if __name__ == '__main__':
    s = SamGen()
    s.gen()