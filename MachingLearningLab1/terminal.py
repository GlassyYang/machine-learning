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
train_num = 10
validate_num = 10
test_num = 10
dim = 7
accuracy = 0.1
lamb = 0.01
alpha = 0.001


# 方法求得拟合得到的函数在某一点上的返回值。w是列向量，是参数系数
def func(x, w):
    res = w[0][0]
    for i in range(1, len(w)):
        for j in range(len(w[i])):
            res += x ** i * w[i][0]
    return res


# 主函数
def main():
    sam_gener = sg.SamGen()
    try:
        f = open(train_file, 'r')
    except FileNotFoundError:
        print('[+]train sample file isn\'t exist, will be generated' )
        sam_gener.set_para(train_file, train_num)
        sam_gener.gen_sample()
    else:
        com = input('[+]train sample file has existed, generate new sample or not?(yes/no)')
        if com == 'yes' or com == 'y' or com == 'Y':
            gen = sg.SamGen()
            gen.set_para(train_file, train_num)
            gen.gen_sample()
        f.close()
    grad_d = gd.GradDecrease(train_file, dim, train_num)
    # an = anal.Analytic(train_file, dim, train_num)
    # w_g1 = grad_d.conj_grad_decrease(accuracy, alpha, lamb)
    # w_g2 = grad_d.grad_decrease(accuracy, alpha, lamb)
    # w_a = an.fit_without_reg()
    # print(w_a)
    # print(w_g)
    # 通过求得的几个参数进行绘图
    plt.xlim((0, 2))
    plt.ylim((-10, 10))
    x1 = np.linspace(0, 2, 50)
    w_g1 = [[-0.0017796803494696065], [-0.0032912866485964153], [-0.005800565771834365], [-0.010865447377433196], [-0.02061698555780289], [-0.03897251494266314], [-0.07319820899063928]]
    # plt.plot(x1, func(x1, w_a), 'r-', linewidth=1, label='conj grad decrease')
    plt.plot(x1, func(x1, w_g1), 'g-', linewidth=1, label="grad decrease")
    # plt.plot(x1, np.sin(2 * pi * x1), 'b-', linewidth=0.8, label="f ' '(x)")

    plt.legend(['f(x)', "f '(x)"], loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()
