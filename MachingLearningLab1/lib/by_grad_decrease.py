#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
    使用梯度下降法求出最优的参数，只考虑加上正则的情况。
'''

from math import fabs
from MachingLearningLab1.lib import mat_oper


class GradDecrease:
    def __init__(self, file, dim, size):
        self.dim = dim
        self.mat_x = []
        self.mat_t = []
        self.size = size
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
    def conj_grad_decrease(self, accuracy, lamb):
        w = []
        for i in range(self.dim):
            w.append([0])
        xtrx = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_x)
        lambda_e = []
        for i in range(self.dim):
            lambda_e.append([])
            # 构建一个单位矩阵乘以lambda / 2的值
            for j in range(self.dim):
                if i == j:
                    lambda_e[i].append(lamb / 2)
                else:
                    lambda_e[i].append(0)
        # 计算出的方程Ax=b中矩阵A的值和矩阵b的值
        # print(len(self.mat_x))
        # print(len(xtrx))
        # print(len(lambda_e))
        mat_a = mat_oper.mat_sum(xtrx, lambda_e)
        b = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_t)
        # 初始化开始时的变量
        aw = mat_oper.mat_multi(mat_a, w)
        r = []
        for i in range(len(b)):
            r.append([])
            for j in range(len(b[0])):
                r[i].append(b[i][j] - aw[i][j])
        p = r
        # 注：k貌似没用
        k = 0
        # 算法的的循环过程
        while True:
            temp = mat_oper.mat_multi(mat_oper.mat_tran(r), r)  # 经检测，语句没毛病
            # print(r)
            # print(mat_oper.mat_tran(r))
            alpha_k = mat_oper.mat_inverse(mat_oper.mat_multi(mat_oper.mat_multi(mat_oper.mat_tran(p), mat_a), p))
            # print(temp)
            # print(alpha_k)
            alpha_k = mat_oper.mat_multi(temp, alpha_k) # 需要注意到经过计算，这儿的alpha k 是一个数，所以后面的数字还需要改一下
            alpha_k = alpha_k[0][0]
            # 由于alpha k 是一个数，所以必须它与向量p的运算就是对向量p中的每一个元素乘以alpha k
            # print(alpha_k)
            for i in range(len(p)):
                for j in range(len(p[i])):
                    p[i][j] = alpha_k * p[i][j]
            w = mat_oper.mat_sum(w, p)
            # 同样，由于alpha是一个数，所以其与矩阵A的乘积也需要另算。
            temp = []
            # print(len(mat_a))
            # print(len(mat_a[0]))
            for i in range(len(mat_a)):
                temp.append([])
                for j in range(len(mat_a[i])):
                    temp[i].append(mat_a[i][j] * alpha_k)
            # print(len(temp))
            # print(len(temp[0]))
            # print(len(p))
            alpha_k_a_p = mat_oper.mat_multi(temp, p)
            r_new = []
            for i in range(len(r)):
                r_new.append([])
                for j in range(len(r[0])):
                    r_new[i].append(r[i][j] - alpha_k_a_p[i][j])
            # 退出循环的条件：if r is sufficiently small, then exit loop.还不知道怎么写。
            temp = 0
            for i in range(len(r_new)):
                for j in range(len(r_new[i])):
                    temp += r_new[i][j]
            print(temp)
            if temp <= accuracy:
                break
            beta = mat_oper.mat_inverse(mat_oper.mat_multi(mat_oper.mat_tran(r), r))
            beta = mat_oper.mat_multi(beta, mat_oper.mat_multi(mat_oper.mat_tran(r_new), r_new))
            # 同上面的alpha，这里的beta同样是一个数，所以需要通过其他的方式计算其与p的乘法
            # print(beta)
            beta = beta[0][0]
            for i in range(len(p)):
                for j in range(len(p[i])):
                    p[i][j] = p[i][j] * beta
            p = mat_oper.mat_sum(r_new, p)
            k = k + 1
        return w
