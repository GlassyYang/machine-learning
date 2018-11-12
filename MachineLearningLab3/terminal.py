import numpy as np
from matplotlib import pyplot as plt
from math import exp, fabs
from time import time
from os.path import exists
import xlrd
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


def data_read_xls(filename):
    data = xlrd.open_workbook(filename)
    table = data.sheet_by_name('Training_Data')
    x = []
    for i in range(1, table.nrows):
        temp = []
        for j in range(1, table.ncols - 4):
            temp.append(float(table.cell(i, j).value))
        x.append(temp)
    return np.array(x)


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
    return classify, center


def em_algorithm(x, k, acc):
    """
    使用em算法分类，初始化为K-算法的输出。
    :param x:   样本点
    :param k:   分类的类数
    ":param acc: 算法退出的条件，当mu在每一轮中每一个元素的变化都小于这个值的时候，算法退出。
    :return: 返回每一个类的参数mu，sigma，以及类别先验。
    """
    init, mu = k_means(x, k)
    total = 0
    for i in init:
        total += len(i)
    p = []
    sigma = []
    for i in range(len(init)):
        p.append(len(init[i]) / total)
        sigma.append(np.eye(len(x[0])))
        mu[i] = np.array(mu[i]).T
    py_x = []
    for i in range(len(x)):
        py_x.append([])
        for j in range(len(mu)):
            py_x[i].append(None)
    while True:
        # E-step
        for i in range(len(py_x)):
            for j in range(len(mu)):
                py_x[i][j] = p[j] * exp(-(x[i] - mu[j]) @ np.linalg.inv(sigma[j]) @ (x[i] - mu[j]).T)
        # m-step
        mu_t = mu[0]
        for i in range(len(mu)):
            deno = 0
            mol1 = np.zeros((1, len(x[0])))
            mol2 = np.zeros((len(x[0]), len(x[0])))
            for j in range(len(py_x)):
                deno += py_x[j][i]
                mol1 += py_x[j][i] * x[j]
                mol2 += py_x[j][i] * (x[j] - mu[i]).T @ (x[j] - mu[i])
            mu[i] = mol1 / deno
            sigma[i] = mol2 / deno
            p[i] = deno / len(x)
        count = 0
        for i in range(len(mu_t[0])):
            if fabs(mu_t[0][i] - mu[0][0][i]) < acc:
                count += 1
        # 算法的退出条件
        if count == len(mu_t):
            break
    # 利用最后得到的概率进行聚类：
    x_class = []
    for i in range(len(mu)):
        x_class.append([])
    for i in range(len(x)):
        maxa = -2
        maxi = -1
        for j in range(len(mu)):
            temp = p[j] * exp(-(x[i] - mu[j]) @ np.linalg.inv(sigma[j]) @ (x[i] - mu[j]).T)
            if temp >= maxa:
                maxa = temp
                maxi = j
        x_class[maxi].append(x[i])
    return p, mu, sigma, x_class


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
    print("menu:\n1.use k-means algorithm\n2.use EM algorithm\ninput a number, 1 or 2")
    com = input(">>>")
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
    if com == '1':
        x_class, mu = k_means(x, 4)
        show(x_class)
    elif com == '2':
        cla, center, s, x_class = em_algorithm(x, 4, 0.04)
        show(x_class)


def diff(filename, x_class):
    """
    输出一个表格，比较iris.data和分类之后的类别中样本点的数量；
    :param filename: 带标签的iris.data数据集
    :param x_class: 由分类算法生成的类
    :return: None
    """
    number = {
        'Iris-setosa': 0,
        'Iris-versicolor': 0,
        'Iris-virginica': 0
    }
    with open(filename, 'r') as f:
        data = f.read()
        lines = data.split('\n')
        lines.pop()
    for i in lines:
        number[i[(i.rfind(',') + 1):len(i)]] += 1
    print('\nresults on data set:')
    print('-------------------------------------')
    print('|', end='')
    for i in number.keys():
        print(i + '\t|', end='')
    print()
    print('|', end='')
    for i in number.keys():
        print(str(number[i]) + '\t\t|', end='')
    print()
    print('-------------------------------------')
    print('results by classify algorithm:')
    print('-------------------------------------')
    for i in range(len(x_class)):
        print(str(len(x_class[i])) + '\t\t|', end='')


def diff_xls(filename, x_class):
    """
    输出一个表格，比较远样本标签和分类之后样本中类数量的区别
    :param filename: 带标签的样本数据
    :param x_class: 有分类算法分得的类
    :return: NNone
    """
    data = xlrd.open_workbook(filename)
    number = {
        'very_low': 0,
        'Low': 0,
        'Middle': 0,
        'High': 0
    }
    sheet = data.sheet_by_name("Training_Data")
    for i in range(1, sheet.nrows):
        number[sheet.cell(i, 5).value] += 1
    print('results on xls data set:')
    print('-------------------------------------')
    print('|', end='')
    for i in number.keys():
        print(i + '\t|', end='')
    print()
    print('|', end='')
    for i in number.keys():
        print(str(number[i]) + '\t\t|', end='')
    print()
    print('-------------------------------------')
    print('results by classify algorithm:')
    print('-------------------------------------')
    for i in range(len(x_class)):
        print(str(len(x_class[i])) + '\t\t|', end='')


def run_uci():
    print('use the xls data set:')
    print('result by k-means algorithm:')
    x = data_read_xls('Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls')
    x_class, mu = k_means(x, 4)
    diff_xls('Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls', x_class)
    p, mu, sigma, x_class = em_algorithm(x, 4, 0.01)
    print('\nresult by EM algorithm:')
    diff_xls('Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls', x_class)
    print("use the second data set:")
    x = data_read('iris_no_label.data')
    x_class, mu = k_means(x, 3)
    diff('iris.data', x_class)
    p, mu, sigma, x_class = em_algorithm(x, 3, 0.005)
    diff('iris.data', x_class)


if __name__ == '__main__':
    # 在测试数据集上跑的结果
    # main()
    # 在uci数据集上跑的结果
    run_uci()
