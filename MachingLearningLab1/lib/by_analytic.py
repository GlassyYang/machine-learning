#! /usr/bin/env python
# -*- coding:UTF-8 -*-
'''
    通过解析式直接求解得到参数，分为加正则和不加正则两种情况，参数的数量需要通过参数传递。
'''
from MachingLearningLab1.lib import mat_oper


# 使用解析式法直接接触最优的值
class Analytic(object):
    def __init__(self, file, dim, size):
        self.mat_x = []
        self.mat_t = []
        self.dim = dim
        self.size = size
        assert self.size != 0
        mat_oper.data_deal(self, file)
        print(self.mat_x)

    # 计算不带正则项的拟合结果
    def fit_without_reg(self):
        xtrx = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_x)
        xtrt = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_t)
        # print(mat_oper.mat_multi(mat_oper.mat_inverse(xtrx), xtrx))
        par_w = mat_oper.mat_multi(mat_oper.mat_inverse(xtrx), xtrt)
        return par_w

    # 计算带正则项的拟合结果.由于带了正则项，所以除了样本容量
    def fit_with_reg(self, lam):
        mat_i = []
        for i in range(self.dim):
            mat_i.append([])
            for j in range(self.dim):
                if i == j:
                    mat_i[i].append(lam)
                else:
                    mat_i[i].append(0)
        xtrx = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_x)
        # for i in range(len(xtrx)):
        #     for j in range(len(xtrx)):
        #         xtrx[i][j] = xtrx[i][j] / self.size
        xtrt = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_t)
        inv = mat_oper.mat_inverse(mat_oper.mat_sum(mat_i, xtrx))
        w = mat_oper.mat_multi(inv, xtrt)
        return w







