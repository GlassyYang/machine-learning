import numpy as np
from time import time
from random import shuffle
from math import e, log
from matplotlib import pyplot as plt
train_file = 'train.txt'
valify_file = 'validate.txt'
abnormal_file = 'abnormal.txt'
dimension = 2
accuracy = .001
alpha = .01
lamb = 0.000001


def sigmod(b):
    """
    calculate and return value of g(x): 1 / (1 + exp(-x)) given b
    :param b: independent variable
    :return: value of g(b)
    """
    return 1 / (1 + e ** (-b))


def data_gen(dim, mu_0, mu_1, sigma_0, sigma_1, len_0, len_1, file):
    """
    generate data set.
    :param dim: dimension of x
    :param mu_0: a list of mus order by x0, ... xdim, having y = 0
    :param mu_1: same as mu_0 but y = 1
    :param sigma_0: list of sigmas having order x0, ... xdim
    :param sigma_1: same as sigma_0 but this list is used by generate data of another class
    :param len_0: number of data having y = 0 which will be generated
    :param len_1: same as len_0, but y = 1
    :param file: output file
    :return: no return
    """
    gen = [[], []]
    np.random.seed(int(time()))
    for i in range(dim):
        gen[0].append(np.random.normal(mu_0[i], sigma_0[i], len_0))
    for j in range(dim):
        gen[1].append(np.random.normal(mu_1[i], sigma_1[i], len_1))
    order = [0] * len_0 + [1] * len_1
    shuffle(order)
    j = k = 0
    data = []
    for i in order:
        if i == 0:
            for l in range(dim):
                data.append(str(gen[0][l][j]))
                data.append(',')
            data.append(str(0))
            j += 1
        else:
            for l in range(dim):
                data.append(str(gen[1][l][k]))
                data.append(',')
            data.append(str(1))
            k += 1
        data.append('\n')
    f = open(file, 'w')
    f.write(''.join(data))
    f.close()


def grad_decrease_reg(dim, x, y, accuracy, alpha, lamb):
    w = np.zeros((dim + 1, 1))
    step = 0
    # m为样本的数量
    m = len(y)
    while True:
        a = x @ w
        shape = a.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                a[i][j] = sigmod(a[i][j])
        temp = alpha / m * (x.T @ (a - y)) + lamb * w
        w = w - temp
        if np.array(np.fabs(temp)).sum() < accuracy:
            break
        step += 1
    return w, step


def grad_decrease(dim, x, y, accuracy, alpha):
    """
    gradient descent method gain minimum w
    :param dim: dimension of parameter w
    :param x: matrix, store value of x
    :param y: vector store value of y
    :param accuracy: define when algorithm stop
    :param alpha: step size
    :return: parameter w and number of iterator
    """
    w = np.zeros((dim + 1, 1))
    step = 0
    while True:
        a = x @ w
        shape = a.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                a[i][j] = sigmod(a[i][j])
        temp = alpha * (x.T @(a - y))
        w = w - temp
        if np.array(np.fabs(temp)).sum() < accuracy:
            break
        step += 1
    return w, step


def data_read(file):
    """
    read data set from file into memory
    :param file: filename which stores data set
    :return: matrix x stores x, matrix y stores y
    """
    x = []
    y = []
    with open(file, 'r') as f:
        data = f.read().split('\n')
    data.pop()
    for line in data:
        arr = line.split(',')
        temp = list()
        temp.append(1)
        for i in range(len(arr) - 1):
            temp.append(float(arr[i]))
        x.append(temp)
        y.append([float(arr[len(arr) - 1])])
    return np.array(x), np.array(y)


def show(x, y, w, w_reg):
    """
    show data set and classifier in the picture.
    :param x: independent variable of data set
    :param y: label of difference classes
    :param w: parameters of a line which distinguish different classes, should be generated without reg
    :param w_reg: same as w, but should be generated with reg
    :return: nothing
    """
    show_x0 = []
    show_y0 = []
    show_x1 = []
    show_y1 = []
    for i in range(len(x)):
        if y[i] == 0:
            show_x0.append(x[i][1])
            show_y0.append(x[i][2])
        else:
            show_x1.append(x[i][1])
            show_y1.append(x[i][2])
    plt.xlim((-5, 10))
    plt.ylim((-5, 10))
    p1 = plt.scatter(show_x0, show_y0, marker='x', color='g', label='dot', s=len(x[0]))
    p1 = plt.scatter(show_x1, show_y1, marker='x', color='b', label='dot', s=len(x[0]))
    x = np.linspace(-5, 10, num=500)
    y = -(w[0][0] + w[1][0] * x) / w[2][0]
    plt.plot(x, y, 'm', linewidth=1, label="without_reg")
    y = -(w_reg[0][0] + w_reg[1][0] * x) / w_reg[2][0]
    plt.plot(x, y, 'c', linewidth=1, label="with_reg")
    plt.legend(['without-reg', "with-reg", "in class 0", "in class 1"], loc='lower right')
    plt.show()


def main(mode):
    f = None
    try:
        if mode == 'normal':
            f = open(train_file, 'r')
        else:
            f = open(abnormal_file, 'r')
    except FileNotFoundError:
        com = input("Cannot find train data, generate it or not?(y/n)")
        if com == 'y':
            mu_0 = [1.4, 1.2]
            mu_1 = [2.65, 5.42]
            sigma = [2.3, 1.3]
            if mode == 'normal':
                data_gen(dimension, mu_0, mu_1, sigma, 40, 32, train_file)
            else:
                sigma_a = [1.2, 2.2]
                data_gen(dimension, mu_0, mu_1, sigma, sigma_a, 40,32, abnormal_file)
        else:
            print("Cannot execute gradient descent under condition of no data set.")
            return
    finally:
        if f is not None:
            f.close()
    if mode == 'normal':
        x, y = data_read(train_file)
    else:
        x, y = data_read(abnormal_file)
    w, step = grad_decrease(dimension, x, y, accuracy, alpha)
    print("parameter theta has generated after %s step by gradient descent without reg" % step)
    w_reg, step = grad_decrease_reg(dimension, x, y, accuracy, alpha, lamb)
    print("parameter theta has generated after %s step by gradient descent with reg" % step)
    show(x, y, w, w_reg)


if __name__ == '__main__':
    print("see result under normal data set or which doesn't satisfy assumption?\n1.normal\n2.abnormal")
    s = input(">>>")
    if s == '1':
        main("normal")
    else:
        main('abnormal')
