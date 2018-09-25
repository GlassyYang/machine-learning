#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
    使用梯度下降法求出最优的参数，只考虑加上正则的情况。
'''

from math import fabs
from MachingLearningLab1.lib import mat_oper
import numpy as np
import numpy.linalg as la


class GradDecrease:
    def __init__(self, file, dim, size):
        self.dim = dim
        self.mat_x = []
        self.mat_t = []
        self.size = size
        mat_oper.data_deal(self, file)

    # 处理传入文件中的数据
    def deal_data(self, file):
        fp = open(file, 'r')
        if not file:
            print("[-]cannot open file " + file)
            exit(1)
        data = fp.read().split('\n')
        # row = [1]
        for line in data:
            temp = line.split('\t')
            self.mat_x.append(float(temp[0]))
            self.mat_t.append(float(temp[1]))
        return

    # 过程中求出的函数h(x),参数为函数系数，返回在某一点的值
    def _h_theta_x(self, omega, x):
        ans = omega[0][0]
        for i in range(1, len(omega)):
            ans += omega[i][0] * x[i]
        return ans

    # 通过梯度下降法求出最优值
    def grad_decrease(self, accuracy, alpha, lamb):  # accuracy: 准确度，alpha：前面的项的系数，lambda：惩罚项的系数
        w = np.zeros((self.dim, 1))
        step = 0
        x = np.array(self.mat_x)
        t = np.array(self.mat_t)
        xtr = x.transpose()
        xtrx = xtr @ x
        xtrt = xtr @ t
        while True:
            temp = alpha * (2 * xtrx @ w - 2 * xtrt + lamb * w)
            w = w - temp
            # 通过计算损失函数判断出终止条件
            temp = np.fabs(temp)
            lost = np.sum(temp)
            if lost < accuracy:
                break
            step += 1
            print(step)
            print(w)
        return w

    # 通过共轭梯度下降法求出最优值
    def conj_grad_decrease(self, accuracy, lamb):
        w = np.zeros((self.dim, 1))
        x = np.array(self.mat_x)
        t = np.array(self.mat_t)
        mat_a = x.transpose() @ x + lamb
        # 计算出的方程Ax=b中矩阵A的值和矩阵b的值
        b = x.transpose() @ t
        # 初始化开始时的变量
        r = np.array(b - mat_a @ w)
        p = r
        # 注：k貌似没用
        k = 0
        # 算法的的循环过程
        while True:
            alpha_k = r.transpose() @ r / (p.transpose() @ mat_a @ p)
            w = w + alpha_k[0][0] * p
            r_new = r - alpha_k[0][0] * mat_a @ p
            #退出条件：
            lost = np.array(np.fabs(r_new)).sum()
            if lost < accuracy:
                break
            print(lost)
            beta = (r_new.transpose() @ r_new) / (r.transpose() @ r)
            p = r_new + beta[0][0] * p
            k = k + 1
            r = r_new
            print(k)
            print(w)
        return w
