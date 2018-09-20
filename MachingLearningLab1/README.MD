>## 程序的运行方式
>程序的入口为terminal.py, 同时还有携带的库lib。  
>需要注意的是，只有在安装了numpy和matplotlib模块的python 3版本上，程序才能正常运行

>## 文件目录及说明
>terminal.py: 程序的执行文件  
>README.md: 程序说明文件  
>lib: 程序的其他库文件  
>lib->am_gen.py: 样本生成器，生成y = sin(2pix)在区间[0,2]上的样本，函数自变量的取值服从(0,2)上的
>均匀分布，同时加上了N(0, 1)的噪声；  
>lib->by_analytic.py: 通过解析式直接求得损失函数的最小值  点
>lib->by_grad_decrease.py: 通过梯度下降法和共轭梯度法求得损失函数的最小值点  
>lib->mat_oper.py: 实现了一些矩阵基本操作的函数，包括转置、逆、乘积、和，从文件中读取数据的函数也在此文件当中实现