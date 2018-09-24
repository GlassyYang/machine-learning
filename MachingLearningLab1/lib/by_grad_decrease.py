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
        w = []
        for i in range(self.dim):   # 系数向量的维度，类似于[[1],[2],[3]]的形式
            w.append([0])
        step = 0
        while True:
            w_new = []
            for i in range(len(w)):
                temp = 0
                for j in range(len(self.mat_x)):
                    a = (self._h_theta_x(w, self.mat_x[j]) - self.mat_t[j][0]) * self.mat_x[j][i]   # 公式(h(xi) - y) * xi
                    temp += a
                if i == 0:
                    temp = alpha * temp + w[i][0]
                else:
                    temp = alpha * temp + lamb * w[i][0]
                temp = temp / len(self.mat_x)
                w_new.append([w[i][0] - temp])
            # 通过计算损失函数判断出终止条件
            lost = 0
            for i in range(len(w)):
                lost += fabs(w[i][0] - w_new[i][0])
            if lost < accuracy:
                w = w_new
                break
            w = w_new
            step = step + 1
            print(step)
            print(w)
        return w

    # 通过共轭梯度下降法求出最优值
    def conj_grad_decrease(self, accuracy, alpha, lamb):
        w = []
        for i in range(self.dim):
            w.append([0])
        xtrx = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_x)
        mat_a = []
        for i in range(len(xtrx)):
            mat_a.append([])
            for j in range(len(xtrx[i])):
                temp = alpha * xtrx[i][j]
                if i == j and i == 0:
                    temp += 1
                elif i == j:
                    temp += lamb
                mat_a[i].append(temp)
        # 计算出的方程Ax=b中矩阵A的值和矩阵b的值
        b = mat_oper.mat_multi(mat_oper.mat_tran(self.mat_x), self.mat_t)
        for i in range(len(b)):
            for j in range(len(b[i])):
                b[i][j] *= alpha
        # 乘以系数alpha
        for i in range(len(xtrx)):
            for j in range(len(xtrx[i])):
                xtrx[i][j] *= alpha
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
            alpha_k = mat_oper.mat_inverse(mat_oper.mat_multi(mat_oper.mat_multi(mat_oper.mat_tran(p), mat_a), p))
            alpha_k = mat_oper.mat_multi(temp, alpha_k) # 需要注意到经过计算，这儿的alpha k 是一个数，所以后面的数字还需要改一下
            alpha_k = alpha_k[0][0]
            # 由于alpha k 是一个数，所以必须它与向量p的运算就是对向量p中的每一个元素乘以alpha k
            for i in range(len(p)):
                for j in range(len(p[i])):
                    p[i][j] = alpha_k * p[i][j]
            w = mat_oper.mat_sum(w, p)
            # 同样，由于alpha是一个数，所以其与矩阵A的乘积也需要另算。
            temp = []
            for i in range(len(mat_a)):
                temp.append([])
                for j in range(len(mat_a[i])):
                    temp[i].append(mat_a[i][j] * alpha_k)
            alpha_k_a_p = mat_oper.mat_multi(temp, p)
            r_new = []
            for i in range(len(r)):
                r_new.append([])
                for j in range(len(r[0])):
                    r_new[i].append(r[i][j] - alpha_k_a_p[i][j])
            # 退出循环的条件：判断损失达到了一个值
            temp = 0
            for i in range(len(self.mat_t)):
                temp += fabs(self._h_theta_x(w, self.mat_x[i]) - self.mat_t[i][0])
            if temp / len(self.mat_t) < accuracy:
                break
            beta = mat_oper.mat_multi(mat_oper.mat_tran(r), r)[0][0]
            beta = mat_oper.mat_multi(mat_oper.mat_tran(r_new), r_new)[0][0] / beta
            # 同上面的alpha，这里的beta同样是一个数，所以需要通过其他的方式计算其与p的乘法
            # print(beta)
            for i in range(len(p)):
                for j in range(len(p[i])):
                    p[i][j] = p[i][j] * beta
            p = mat_oper.mat_sum(r_new, p)
            k = k + 1
            print(k)
            print(w)
        return w
