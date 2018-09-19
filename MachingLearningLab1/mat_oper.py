#! /usr/bin/env python
# -*- coding: UTF-8 -*-
'''
    定义了几个求解矩阵乘积和矩阵逆的算法。mat_muti()用于求矩阵的乘积，mat_inverse()用于求矩阵的逆，
    使用的方法是伴随矩阵法。mat_tran()求解矩阵的转置并返回，mat_sum()用于求解两个矩阵的和。所有这些
    函数都不会改变传递的参数矩阵的值，结果会成功函数中返回。
'''


# 从生成的数据文件中读出数据并转换成数据
def data_deal(obj, file,):
    fp = open(file, 'r')
    if not fp:
        print("[-]Error! file named" + file + "doesn't exist")
        exit(1)
    data = fp.read().split('\n')
    for s in  data:
        temp = s.split('\t');
        # 验证数据的正确性
        assert len(temp) == 2 and isinstance(temp[0], float) and isinstance(temp[1], float)
        assert temp[0].isalnum() and temp[1].isalnum()
        row = [1]
        x = float(temp[0])
        for v in temp:
            row.append[row[v - 1] * x]
        obj.mat_x.append(row)
        obj.mat_t.append([float(temp[1])])
    return


# 计算两个矩阵的乘法
def mat_multi(x, y):
    ans = []
    for k in range(len(x)):
        row = []
        for i in range(len(y[0])):
            temp = 0
            for j in range(len(y)):
                temp += x[k][j] * y[j][i]
            row.append(temp)
        ans.append(row)
    return ans


# 计算矩阵的逆，要求传入的矩阵必须是方阵.当矩阵可逆时，返回其逆矩阵；如果不可逆，则返回None
def mat_inverse(x):
    e = []
    # 拷贝传入参数
    x_cp = []
    for i in range(len(x)):
        x_cp.append([])
        for j in range(len(x)):
            x_cp[i].append(x[i][j])
    # 构造单位矩阵
    for i in range(len(x)):
        e.append([])
        for j in range(len(x)):
            if i == j:
                e[i].append(1)
            else:
                e[i].append(0)
    # 利用伴随矩阵法计算矩阵的逆，要求矩阵必须可逆。
    for i in range(len(x)):
        temp = 1 / x_cp[i][i]
        for j in range(i, len(x)):
            x_cp[i][j] = x_cp[i][j] * temp
        for j in range(len(x)):
            e[i][j] = e[i][j] * temp
        for j in range(len(x)):
            if i != j and x_cp[j][i] != 0:
                temp = - x_cp[j][i]
                for k in range(i, len(x)):
                    x_cp[j][k] = x_cp[j][k] + temp * x_cp[i][k]
                for k in range(len(x)):
                    e[j][k] = e[j][k] + temp * e[i][k]
    for i in range(len(x_cp)):
        if x_cp[i][i] == 0:
            return None
    return e


# 函数计算矩阵的转置并返回
def mat_tran(x):
    for i in range(len(x)):
        for j in range(i, len(x)):
            temp = x[i][j]
            x[i][j] = x[j][i]
            x[j][i] = temp


# 计算两个矩阵的和
def mat_sum(x, y):
    m_sum = []
    for i in range(len(x)):
        m_sum.append([])
        for j in range(len(x[0])):
            m_sum[i].append(x[i][j] + y[i][j])
    return m_sum


# 测试代码
if __name__ == '__main__':
    test1 = [[1, 1], [0, 1], [0, 0]]
    test2 = [[1, 2, 3], [2, 1, 2]]
    print(test1)
    print(mat_inverse(test1))   # 测试求逆算法是否正确
    print(mat_multi(test1, test2))   # 测试矩阵的乘积算法是否正确
