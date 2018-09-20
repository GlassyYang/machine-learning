#! /usr/bin/env python
# -*- coding:UTF-8 -*-
'''
    运行主程序。
'''

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from MachingLearningLab1.lib import sam_gen as sg, by_grad_decrease as gd, by_analytic as anal
# 定义训练数据、验证数据和测试数据所在的文件
train_file = 'train_sample.txt'
vali_file = 'validate_sample.txt'
test_file = 'test_file.txt'
# 定义各个样本中数据的个数
train_num = 7
validate_num = 2
test_num = 1


# 方法求得拟合得到的函数在某一点上的返回值。w是列向量，是参数系数
def func(x, w):
    res = w[0][0]
    x_buf = x
    for i in range(1, len(w)):
        for j in range(len(w[i])):
            res += x_buf * w[i][j]
            x_buf *= x
    # print(res)
    return res


# 主函数
def main():
    sam_gener = sg.SamGen()
    try:
        f = open(test_file, 'r')
    except FileNotFoundError:
        print('[+]train sample file isn\'t exist, will be generated' )
        sam_gener.set_para(train_file, train_num)
        sam_gener.gen_sample()
    else:
        # com = input('[+]train sample file has existed, generate new sample or not?(yes/no)')
        # if com == 'yes' or com == 'y' or com == 'Y':
            # sg.SamGen().gen()
        f.close()
    grad_d = gd.GradDecrease(train_file, 5, train_num)
    an = anal.Analytic(train_file, 5, train_num)
    w_g = grad_d.conj_grad_decrease(0.001, 0.01)
    w_a = an.fit_without_reg()
    def funct(x):
        res = w_a[0][0]
        x_buf = x
        for i in range(1, len(w_a)):
            for j in range(len(w_a[i])):
                res += x_buf * w_a[i][j]
                x_buf *= x
        # print(res)
        return res
    # print(w_a)
    # print(w_g)
    # 通过求得的几个参数进行绘图
    plt.xlim((0, 2))
    plt.ylim((-10, 10))
    x1 = np.linspace(0, 2, 100)
    plt.plot(x1, funct(x1), 'r-', linewidth=1, label='f(x)')
    # plt.plot(x1, func(x1, w_g), 'g-', linewidth=1, label="f '(x)")
    # plt.plot(x1, np.sin(2 * pi * x1), 'b-', linewidth=0.8, label="f ' '(x)")

    plt.legend(['f(x)', "f '(x)", "f ' '(x)"], loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
