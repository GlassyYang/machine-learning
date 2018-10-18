import numpy as np
from time import time
from random import shuffle

def data_gen(dim, mu_0, mu_1, sigma, len_0, len_1, file):
    gen = [[], []]
    np.random.seed(int(time()))
    for i in range(dim):
        gen[0].append(np.random.normal(mu_0[i], sigma[i], len_0))
    for j in range(dim):
        gen[1].append(np.random.normal(mu_1[i], sigma[i], len_1))
    order = [0] * len_0 + [1] * len_1
    shuffle(order)
    j = k = 0
    data = []
    for i in order:
        if i == 0:
            for l in range(dim):
                data.append(str(gen[0][l][j]))
                data.append(',')
            j += 1
            data.append(str(0))
            data.append('\n')
        else:
            for l in range(dim):
                data.append(str(gen[1][l][k]))
                data.append(',')
            data.append(str(1))
            data.append('\n')
            k += 1
        data.append('\n')
    f = open(file, 'w')
    f.write(''.join(data))
    f.close()

