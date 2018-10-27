import numpy as np
from matplotlib import pyplot as plt
from time import time
from os.path import exists
file = 'data.txt'


def data_gen(dim, mu, sigma, length, file):
    """
    generate data set.
    :param dim: dimension of x
    :param mu: a list of mus order by x0, ... xdim,
    :param sigma: a list of sigmas having order x0, ... xdim
    :param length: number of data having y = 0 which will be generated
    :param file: a file pointer, and can write data into this file ptr.
    :return: no return
    """
    gen = []
    np.random.seed(int(time()))
    for i in range(dim):
        gen.append(np.random.normal(mu[i], sigma[i], length))
    lines = []
    for i in range(length):
        line = []
        for j in range(dim):
            line.append(str(gen[j][i]))
        lines.append(','.join(line) + '\n')
    file.writelines(lines)


def min_dist(x, c):
    """
    Given a vector x and a list of center dots, return index of center which has minimum distance with x
    :param x: vector x
    :param c: list of center dots.
    :return: index of center dots of passed list, which has minimum distance of x among center dots.
    """
    cur_min = 0
    for i in range(len(c[0])):
        cur_min += (x[i] - c[0][i]) ** 2
    mini = 0
    for i in range(1, len(c)):
        temp = 0
        for j in range(len(c[i])):
            temp += (x[j] - c[i][j]) ** 2
        if temp < cur_min:
            cur_min = temp
            mini = i
    return mini


def cal_center(cla):
    """
    Given a class, calculate center dot of nonempty class.
    :param cla: a class.
    :return: center dot of the class. But if the class is empty, return None.
    """
    if len(cla) == 0:
        return None
    center = []
    for i in range(len(cla[0])):
        center.append([])
        center[i].append(cla[0][i])
    for i in range(len(cla)):
        for j in range(len(cla[i])):
            center[j] += cla[i][j]
    for i in range(len(center)):
        center[i] /= len(cla)
    return center


def k_means(x, k):
    """
    k means algorithm, inout k and a data matrix x, output list of classes classified from x.
    :param x: data matrix x
    :param k: number of classes.
    :return: a list of classes, each data from x will be classified into one of them
    """
    # 确定地选择初始的样本中心
    # center = [[3, 3], [3, 6.5], [6.5, 3], [6.5, 6.5]]
    center = []
    classify = []
    # initialize
    for i in range(k):
        # 随机地从样本点中选取四个作为初始的样本中心。
        c = np.random.randint(len(x))
        center.append(x[c])
        classify.append([])
    for i in range(len(x)):
        classify[min_dist(x[i], center)].append(x[i])
    i = 0
    center.clear()
    while i < len(classify):
        temp = cal_center(classify[i])
        if temp is not None:
            center.append(temp)
            i += 1
        else:
            del classify[i]
    del i
    while True:
        count = 0
        new_class = []
        for i in range(len(center)):
            new_class.append([])
        # 更新分类中的样本点
        for i in range(len(center)):
            for j in range(len(classify[i])):
                index = min_dist(classify[i][j], center)
                if index != i:
                    count += 1
                new_class[index].append(classify[i][j])
        del classify
        classify = new_class
        # 如果更新样本点的类别的过程中没有改变样本点的类别，就退出循环
        if count == 0:
            break
        # 计算新的样本中心
        center.clear()
        i = 0
        while i < len(classify):
            temp = cal_center(classify[i])
            if temp is not None:
                center.append(temp)
                i += 1
            else:
                del classify[i]
        del i
    return classify


def show(cla):
    """
    Given a list of different class, show them in the window with difference color
    :param cla: a list of class
    :return: None
    """
    color = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
    if len(cla) > len(color):
        print('Too classes given!')
        return
    plt.xlim((0, 10))
    plt.ylim((0, 10))
    for i, clo in zip(cla, color):
        show_x = []
        show_y = []
        for j in i:
            show_x.append(j[0])
            show_y.append(j[1])
        plt.scatter(show_x, show_y, marker='x', color=clo, label='dot', s=len(show_x))
    plt.show()


def data_read(filename):
    """
    read data set from file into memory
    :param filename: filename which stores data set
    :return: matrix x stores x
    """
    x = []
    with open(filename, 'r') as f:
        data = f.read().split('\n')
    data.pop()
    for line in data:
        arr = line.split(',')
        temp = list()
        for i in range(len(arr)):
            temp.append(float(arr[i]))
        x.append(temp)
    return np.array(x)


def main():
    global file
    if not exists(file):
        mu_0 = [3, 3]
        mu_1 = [6.5, 3]
        mu_2 = [3, 6.5]
        mu_3 = [6.5, 6.5]
        sigma_0 = [0.5, 1]
        sigma_1 = [.6, 1.1]
        sigma_2 = [1.2, .7]
        sigma_3 = [.5, .6]
        with open(file, 'w') as f:
            data_gen(2, mu_0, sigma_0, 13, f)
            data_gen(2, mu_1, sigma_1, 9, f)
            data_gen(2, mu_2, sigma_2, 10, f)
            data_gen(2, mu_3, sigma_3, 11, f)
    x = data_read(file)
    show([x])
    cla = k_means(x, 4)
    show(cla)


if __name__ == '__main__':
    main()
