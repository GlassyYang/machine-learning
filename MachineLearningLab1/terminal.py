#! /usr/bin/env python
# -*- coding:UTF-8 -*-
'''
    运行主程序。
'''

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from MachineLearningLab1.lib import sam_gen as sg, by_grad_decrease as gd, by_analytic as anal
import re
from math import fabs
# 定义训练数据、验证数据和测试数据所在的文件
train_file = 'train_sample.txt'
vali_file = 'validate_sample.txt'
test_file = 'test_file.txt'
# 定义各个样本中数据的个数
train_num = 70
validate_num = 20
test_num = 10


# 方法求得拟合得到的函数在某一点上的返回值。w是列向量，是参数系数
def func(x, w):
    res = w[0][0]
    for i in range(1, len(w)):
        for j in range(len(w[i])):
            res += x ** i * w[i][0]
    return res


def validate(file, w):
    fit = 0
    sum = 0
    with open(file, 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            temp = line.split('\t')
            if len(temp) != 2:
                break
            x = float(temp[0])
            y = float(temp[1])
            if fabs(func(x, w) - y) < 1e-3:
                fit += 1
            sum += 1
    return fit, sum


def gen_sample():
    sam_gener = sg.SamGen()
    try:
        f = open(train_file, 'r')
    except FileNotFoundError:
        print('[+]sample file isn\'t exist, will be generated')
        sam_gener.set_para(train_file, train_num)
        sam_gener.gen_sample()
        sam_gener.set_para(vali_file, validate_num)
        sam_gener.gen_sample()
        sam_gener.set_para(test_file, test_num)
        sam_gener.gen_sample()
    else:
        f.close()
        com = input('[+]train sample file has existed, generate new sample or not?(yes/no)')
        if com == 'yes' or com == 'y' or com == 'Y':
            sam_gener.set_para(train_file, train_num)
            sam_gener.gen_sample()
            sam_gener.set_para(vali_file, validate_num)
            sam_gener.gen_sample()
            sam_gener.set_para(test_file, test_num)
            sam_gener.gen_sample()


# 测试函数
def test():
    dim = 7
    accuracy = 0.000001
    lamb = 0.01
    alpha = .01
    grad_d = gd.GradDecrease(train_file, dim, train_num)
    # an = anal.Analytic(train_file, dim, train_num)
    # w_g1 = grad_d.conj_grad_decrease(accuracy, lamb)[0]
    w_g2 = grad_d.grad_decrease(accuracy, alpha, lamb)[0]
    # w_a1 = an.fit_without_reg()
    # w_a2 = an.fit_with_reg(lamb)
    # 通过求得的几个参数进行绘图
    plt.xlim((-1, 1))
    plt.ylim((-5, 5))
    x1 = np.linspace(-1, 1, 100)
    # plt.plot(x1, func(x1, w_g1), 'r-', linewidth=1, label='conj-grad-decrease')
    plt.plot(x1, func(x1, w_g2), 'c', linewidth=1, label="grad-decrease")
    # plt.plot(x1, func(x1, w_a1), 'm', linewidth=1, label='func-without-reg')
    # plt.plot(x1, func(x1, w_a2), 'pink', linewidth=1, label="func-with-reg")
    plt.plot(x1, np.sin(2 * pi * x1), 'c', linewidth=0.8, label="sinx' '(x)")

    plt.legend(['analize-without-reg', "analize-with-reg", "sin(x)", "sample"], loc='lower right')
    plt.show()


# 改变梯度下降法的学习率，比较学习率对循环步数的影响
def alpha_change():
    alpha = (1, 0.1, 0.01, 0.001)
    steps = []
    lam = 0.1
    accuracy = 0.000001
    dim = 7
    grad = gd.GradDecrease(train_file, dim, train_num)
    for i in alpha:
        steps.append(grad.grad_decrease(accuracy, i, lam)[1])
    print("experiment takes unber condition: accuracy: %s, lambda: %s, dimension: %s" % (accuracy, lam, dim))
    print("alpha\t|\t" + re.subn("\(|\)", "", str(alpha))[0].replace(',', '\t|\t'))
    print("step\t|\t" + re.subn("\[|\]", "", str(steps))[0].replace(",", '\t|\t'))


def lambda_change():
    lam = [1000, 1, 0.0001, 0.0000001]
    accuracy = .000001
    dim = 10
    file = "train_overfiting.txt"
    num = 10
    gen = sg.SamGen()
    gen.set_para(file, num)
    gen.gen_sample()
    grad = gd.GradDecrease(file, dim, num)
    train_result = []
    vali_result = []
    for i in lam:
        w = grad.conj_grad_decrease(accuracy, i)[0]
        x1 = np.linspace(-1, 1, 100)
        plt.plot(x1, func(x1, w), 'm', linewidth=1, label='func-without-reg')
        plt.show()
        train_result.append(re.subn("\(|\)", '', str(validate(file, w)))[0].replace(',', '/'))
        vali_result.append(re.subn("\(|\)", '', str(validate(vali_file, w)))[0].replace(',', '/'))
    print("experiment takes unber condition:")
    print("numbers of sample:%s, dimension: %s, algorithm: conj_decrease, accuracy: %s" % (num, dim, accuracy))
    print("lambda\t\t\t\t\t\t|\t" + re.subn("\[|\]", "", str(lam))[0].replace(',', '\t\t|\t'))
    print("fit/sum on train sample\t\t|\t" + re.subn("\[|\]|\'", "", str(train_result))[0].replace(',', "\t|\t"))
    print("fit/sum on validate sample\t|\t" + re.subn("\[|\]|\'", "", str(vali_result))[0].replace(',', "\t|\t"))


if __name__ == '__main__':
    # test()
    alpha_change()
    # lambda_change()
