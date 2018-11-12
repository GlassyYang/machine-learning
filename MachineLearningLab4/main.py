import numpy as np
from os.path import exists
from PIL import Image
import struct
from math import fabs


def data_gen(filename, num):
    """
    产生数据并将其写入到文件，产生的数据为三维的数据，其中两维数据的取值是相同的（但是在相同的基础上加上了噪声），另外
    两维的数据服从二维的高斯分布。
    :param filename: 存储数据的文件
    :param num: 生成样本点的数量
    :return: 无
    """
    data = np.random.randn(num, 2)
    noise = np.random.normal(0, 0.1, num)
    line = '%f %f %f\n'
    lines = []
    for i in range(num):
        lines.append(line % (data[i][0], data[i][0] + noise[i], data[i][1]))
    with open(filename, 'w') as f:
        f.writelines(lines)
    return


def data_read(filename):
    """
    从文件中读取数据，并返回一个数组，数组的元素都是一个ndarray对象，保存着每一个数据，数据形状为(1, 3)
    :param filename:
    :return:
    """
    with open(filename, 'r') as f:
        data = f.read()
        data = data.split('\n')
        if data[-1] == '':
            data.pop()
    x = []
    for i in data:
        temp = []
        ltemp = i.split(' ')
        for j in ltemp:
            temp.append([float(j)])
        x.append(np.array(temp))
    return x


def pca(x):
    """
    使用pca算法表示对输入的数据进行降维
    :param x: 一个list，其中的每一个元素都是一个ndarray对象，用于表示一个样本点，是列向量。
    :return: 返回学习得到的特征向量、失真值，以及原来的x在新向量下的表示
    """
    x_ave = x[0]
    for i in range(1, len(x)):
        x_ave += x[i]
    x_ave /= len(x)
    s = (x[0] - x_ave) @ (x[0] - x_ave).T
    for i in range(1, len(x)):
        s += (x[i] - x_ave) @ (x[i] - x_ave).T
    s /= len(x)
    # 使用numpy中的库计算S的特征值和特征向量
    lamb, mu_list = np.linalg.eig(s)
    # 去掉最小的特征值对应的特征向量
    b = lamb[0]
    mini = 0
    for i in range(1, len(mu_list)):
        if b > lamb[i]:
            b = lamb[i]
            mini = i
    print("find minimum eigenvalue: %f" % b)
    # 删除最小特征值对应的特征向量
    temp = []
    for i in range(len(mu_list)):
        if i != mini:
            temp.append(mu_list[:, i])
    mu_list = np.array(temp)
    # 为每一个x，计算其在当前特征向量下的表示
    x_rep = []
    for i in x:
        temp = []
        for j in mu_list:
            temp.append([i.T @ j])
        x_rep.append(np.array(temp))
    return mu_list, b, x_rep


def store_result(filename, mu_list, b, x_rep):
    """
    将pca算法得到的结果存储到给出的文件中，将会覆盖原文件中的内容
    :param filename: 要将数据写入的文件f
    :param mu_list: 一个列表，存储着得到的特征向量
    :param b: 失真值
    :param x_rep: 原来的数据在新的特征向量下的表示
    :return: 无
    """
    container = list(["eigenvector: \n", ])
    for i in mu_list:
        container.append(str(i) + '\n')
    container.append("distortion: %f\n" % b)
    container.append('representation of each sample point:\n')
    for i in x_rep:
        temp = []
        for j in i.T[0]:
            temp.append(str(j))
        temp = ' '.join(temp)
        container.append(temp + '\n')
    with open(filename, 'w') as f:
        f.writelines(container)
    return


def mnist_read(filename):
    x = []
    with open(filename, 'rb') as f:
        data = f.read(16)
        row = data[11]
        colum = data[15]
        while True:
            data = f.read(row * colum)
            if data == b'':
                break
            x_i = []
            for i in range(len(data)):
                x_i.append([data[i]])
            x.append(np.array(x_i))
    return x, row, colum


def mnist_pca(x):
    """
    使用pca算法对输入的minst数据集中的图像进行降维
    :param x: 一个list，其中的每一个元素都是一个ndarray对象，用于表示一个图片，是列向量。
    :return: 返回学习得到的特征向量、失真值，以及原来的x在新向量下的表示
    """
    x_ave = np.zeros(x[0].shape)
    for i in range(1, len(x)):
        x_ave += x[i]
    x_ave = np.true_divide(x_ave, len(x))
    s = np.zeros((len(x[0]), len(x[0])))
    for i in range(1, len(x)):
        s += (x[i] - x_ave) @ (x[i] - x_ave).T
    s /= len(x)
    # 使用numpy中的库计算S的特征值和特征向量
    lamb, mu_list = np.linalg.eig(s)
    # 从得到的向量集中删除百分之10的最小的向量。
    rmov = []
    b = 0
    count = 0   # 记录从特征向量中删除掉的向量的个数。
    # 进行特征值从大到小的排序，选取
    max_n = max(lamb)
    for i in range(int(len(lamb) * 0.9)):
        min_n = lamb[0]
        mini = 0
        for j in range(len(lamb)):
            if min_n > lamb[j]:
                min_n = lamb[j]
                mini = j
        rmov.append(mini)
        lamb[mini] = max_n
        b += min_n
        count += 1
    print("number of removed feature vector is %d\n" % count)
    # 删除特征值存在于待删除的列表中的特征向量
    muti = np.delete(mu_list, rmov, axis=1)
    # 为每一个x，计算其在当前特征向量下的表示
    mnist_show(x, 28, 28, True)
    x_rep = []
    for i in x:
        temp = (muti.T @ i)
        x_rep.append(muti @ temp)
    return mu_list, b, x_rep


def mnist_show(x, row, colum, flag):
    """
    从读入的数据中选取10幅照片，将其分解成单独的文件保到nmist目录下，用于图像前后的变化的观察
    :param x: 从文件中读取的数据集
    :param row: 图像的行数
    :param colum: 图像的列数
    :param flag: False表示集合x数未处理之前的x，True表示处理之后的x。两者唯一的区别就是存储的文件路径会不一样。
    :return:无
    """
    route = './mnist/'
    if flag:
        route += 'ori_fig/'
    else:
        route += 'res_fig/'
    filename = '%d.bmp'
    for i in range(10):
        arr = x[i].astype(np.uint8).reshape(row, colum)
        img = Image.fromarray(arr)
        filepath = route + (filename % i)
        img.save(filepath)
    return


def mnist_trun(i_filename, o_filename):
    """
    截取初始的十幅图像进行测试
    :param i_filename: 源数据集文件
    :param o_filename: 数据输出的文件
    :return:
    """
    ifile = open(i_filename, 'rb')
    ofile = open(o_filename, 'wb')
    data = ifile.read(16)
    magic_number, img_number, height, width = struct.unpack_from('>IIII', data, 0)
    data = struct.pack(">IIII", magic_number, 10, height, width)
    ofile.write(data)
    size = height * width
    for i in range(10):
        data = ifile.read(size)
        ofile.write(data)
    ifile.close()
    ofile.close()


def main():
    print("choice mode:\n1. run test sample\n2.run mnist data set\n")
    # com = input("input number of the mode you choice:")
    com = 2
    if com == 1:
        sample_file = 'sample.txt'
        result_file = 'result.txt'
        if not exists(sample_file):
            print("no sample file exists, will generate sample points into new sample file...")
            data_gen(sample_file, 50)
        x = data_read(sample_file)
        mu_list, b, x_rep = pca(x)
        store_result(result_file, mu_list, b, x_rep)
    elif com == 2:
        sample_file = './mnist/train-images.idx3-ubyte'
        trun_file = './mnist/train-images.idx3-ubyte-trun'
        if not exists(sample_file):
            print("cannot find mnist data set file!\n")
            return
        if not exists(trun_file):
            mnist_trun(sample_file, trun_file)
        x, row, colum = mnist_read(trun_file)
        # mnist_show(x, row, colum, True)
        mu_list, d, x_rep = mnist_pca(x)
        # 将x中的复数全部转为实数
        trans = []
        for i in x_rep:
            temp = []
            for j in range(len(i)):
                temp.append([])
                for k in range(len(i[j])):
                    temp[j].append(fabs(i[j][k].real))
            trans.append(np.array(temp))
        x_rep = trans
        mnist_show(x_rep, row, colum, False)
    return


if __name__ == '__main__':
    main()
