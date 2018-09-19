#! /usr/bin/env python
# -*- coding:UTF-8 -*-
'''
    通过解析式直接求解得到参数，分为加正则和不加正则两种情况，参数的数量需要通过参数传递。
'''
from . import mat_oper


# 使用解析式法直接接触最优的值
class Analytic(object):
    def __init__(self, file, dim):
        self.mat_x = []
        self.mat_t = []
        self.dim = dim
        mat_oper.data_deal(self, file)

    # 计算不带正则项的拟合结果
    def fit_without_reg(self):
        xtrx = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_x)
        xtrt = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_t)
        par_w = mat_oper.mat_multi(mat_oper.mat_inverse(xtrx, xtrt))
        return par_w

    # 计算带正则项的拟合结果
    def fit_with_reg(self, lam):
        mat_i = []
        for i in range(self.dim):
            mat_i.append([lam])
        xtrx = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_x)
        xtrt = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_t)
        inv = mat_oper.mat_inverse(mat_oper.mat_sum(mat_i, xtrx))
        w = mat_oper.mat_multi(xtrt, inv)
        return w







