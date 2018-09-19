#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
    使用梯度下降法求出最优的参数，只考虑加上正则的情况。
'''

from math import fabs
from . import mat_oper


class GradDecrease:
    def __init__(self, dim, file):
        self.dim = dim
        self.mat_x = []
        self.mat_t = []
        mat_oper.data_deal(self, file)

    # 处理传入文件中的数据
    def deal_data(self,file):
        fp = open(file, 'r')
        if not file:
            print("[-]cannot open file " + file)
            exit(1)
        data = fp.read().split('\n')
        row = [1]
        for line in data:
            temp = line.split('\t')
            self.mat_x.append(float(temp[0]))
            self.mat_t.append(float(temp[1]))
        return

    # 过程中求出的函数h(x),参数为函数系数，返回在某一点的值
    def _h_theta_x(self, omega, x):
        ans = omega[0]
        for i in range(1, len(omega)):
            ans += omega[i] * x[i]
        return ans

    # 通过梯度下降法求出最优值
    def grad_decrease(self, accuracy, alpha, lamb):
        w = []
        for i in range(self.dim):
            w.append(0)
        ch = 100
        while ch < accuracy:
            for i in range(len(w)):
                temp = 0
                for j in range(len(self.mat_x)):
                    a = (self._h_theta_x(w, self.mat_x[j]) - self.mat_t[j] * self.mat_x[i + 1])
                    temp += alpha * (a + lamb * w[i])
                w[i] = w[i] - temp
            ch = fabs(ch - self._h_theta_x(w, self.mat_x[0]))
        return w

    # 通过共轭梯度下降法求出最优值
    def conj_grad_decrease(self, accuracy, alpha, lamb):
        a = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_x)
        ax0 = mat_oper.mat_multi(a, self.mat_x[0])
        b = mat_oper.mat_multi(mat_oper.mat_tran(self.x), self.mat_t)
        for i in range(len(ax0)):
            ax0[i][0] = - ax0[i][0]
        r = mat_oper.mat_sum(self.mat_t, ax0)


