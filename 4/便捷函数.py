# 知识
# 4.2 动手实践：股票相关性分析
# 协方差描述的是两个变量共同变化的趋势，其实就是归一化前的相关系数.
# cov函数：计算协方差矩阵。
# diagonal函数：查看对角线上的元素。
# trace函数：计算矩阵的迹，即对角线上元素之和。
# std函数：标准差。
# 两个向量的相关系数被定义为：协方差（cov）除以各自标准差（std）的乘积。
# corrcoef函数：计算相关系数，相关系数矩阵。
# abs函数：绝对值。
#
# 4.4 动手实践：多项式拟合
# polyfit函数：对数据进行了多项式拟合；
# polyval函数：计算多项式的取值；
# roots函数：求得多项式函数的根；
# polyder函数：求导，求解多项式函数的导函数。
#
# 4.5 净额成交量
# 由当日收盘价、前一天的收盘价以及当日成交量计算得出.
# sign函数：返回数组中每个元素的正负符号，数组元素为负时返回-1，为正时返回1，否则返回0。
# piecewise函数：获取数组元素的正负；可以分段给定取值，使用合适的返回值和对应的条件调用该函数；
# p.piecewise(change, [change < 0, change > 0], [-1, 1])
# array_equal函数：检查数值是否一致相同。
#
# 4.7 交易过程模拟，避免使用循环
# vectorize函数：可以减少你的程序中使用循环的次数，相当于Python中的map函数。
# map函数：会根据提供的函数对指定序列做映射;返回迭代器。
# 第一个参数function以参数序列中的每一个元素调用function函数，返回包含每次function函数返回值的新列表。
# 见下4.7例题.
# round函数：返回浮点数x的四舍五入值。round( x, n)：x：表达式/数值，n：小数点精度。
#
# 4.9 数据平滑
# 4.10 动手实践：使用 hanning 函数平滑数据
# hanning函数：对股票收益率数组进行了平滑处理；
# convolve函数:计算一组数据与指定权重的卷积；
# polysub函数：对两个多项式作差运算；
# isreal函数：判断数组元素是否为实数；
# select函数：可以根据一组给定的条件，从一组元素中挑选出符合条件的元素并返回数组；
# trim_zeros函数：去掉数组首尾的0元素。
# hamming、blackman、bartlett以及kaiser简单使用，见下面4.10例题.


# 例题
# 4.2 动手实践：股票相关性分析
import numpy as np
import matplotlib.pyplot as plt

bhp = np.loadtxt('BHP.csv', delimiter=',', usecols=(6,), unpack=True)
vale = np.loadtxt('VALE.csv', delimiter=',', usecols=(6,), unpack=True)
bhp_returns = np.diff(bhp) / bhp[:1]
vale_returns = np.diff(vale) / vale[:1]
covariance = np.cov(bhp_returns, vale_returns)  # cov：计算协方差矩阵。
print('协方差：', covariance)
print('协方差对角线元素：', covariance.diagonal())  # diagonal：查看对角线上的元素。
print('协方差对角线元素和：', covariance.trace())  # trace：计算矩阵的迹，即对角线上元素之和。
print('两个向量的相关系数:', covariance / (bhp_returns.std() * vale_returns.std()))  # std：标准差。
print('相关系数:', np.corrcoef(bhp_returns, vale_returns))  # corrcoef:计算相关系数，相关系数矩阵.
difference = bhp - vale
avg = np.mean(difference)
dev = np.std(difference)
print('不同步：', np.abs(difference[-1] - avg) > 2 * dev)
t = np.arange(len(bhp_returns))
plt.plot(t, bhp_returns, lw=1)
plt.plot(t, vale_returns, lw=2)
plt.legend(loc='best', labels=['bhp', 'vale'])
plt.show()

# 4.4 动手实践：多项式拟合
import numpy as np
import matplotlib.pyplot as plt
import sys

bhp = np.loadtxt('BHP.csv', delimiter=',', usecols=(6,), unpack=True)
vale = np.loadtxt('VALE.csv', delimiter=',', usecols=(6,), unpack=True)
t = np.arange(len(bhp))
poly = np.polyfit(t, bhp-vale, int(sys.argv[1]))  # polyfit：对数据进行了多项式拟合
print('数据进行了多项式拟合:', poly)
print('下一个值：', np.polyval(poly, t[-1] + 1))  # polyval：计算多项式的取值；
print('多项式的根：', np.roots(poly))
der = np.polyder(poly)
print('导数求导：', der)
print('极值：', np.roots(der))
vals = np.polyval(poly, t)
print('极大值：', np.argmax(vals))
print('极小值：', np.argmin(vals))

plt.plot(t, bhp - vale)
plt.plot(t, vals)
plt.show()

# 4.5 净额成交量
import numpy as np

c, v = np.loadtxt('阿里股票价格变化.csv', delimiter=',', usecols=(1, 2), unpack=True)
change = np.diff(c)
print('收盘价变化量：', change)

signs = np.sign(change)  # sign:返回正、负、0.
print('收盘价变化量正负号：', signs)

pieces = np.piecewise(change, [change < 0, change > 0], [-1, 1])
print('收盘价变化量正负号：', pieces)  # piecewise：返回指定的数（正负)形式.

print('阵列相等:', np.array_equal(signs, pieces))  # array_equal：检查数值是否一致相同。
print('净额成交量：', v[1:] * signs)

# 4.7 交易过程模拟，避免使用循环
import numpy as np
import sys

o, h, l, c = np.loadtxt('阿里股票价格变化.csv', delimiter=',', usecols=(3, 4, 5, 1), unpack=True)

def calc_profit(open, high, low, close):
    # 在开盘时买入
    buy = open * float(sys.argv[1])
    # 当日股价区间
    if low < buy < high:
        return (close - buy)/buy
    else:
        return 0

func = np.vectorize(calc_profit)
profits = func(o, h, l, c)
print('profits(利润）：', profits)

# 非零利润的交易日平均利润
real_trades = profits[profits != 0]
print('交易数量：', len(real_trades), round(100.0 * len(real_trades)/len(c), 2), "%")
print('平均损益%：', round(np.mean(real_trades) * 100, 2))

# 正盈利的交易日平均利润
winning_trades = profits[profits > 0]
print('盈利交易数量：', len(winning_trades), round(100.0 * len(winning_trades)/len(c), 2), "%")
print('平均利润%：', round(np.mean(winning_trades) * 100, 2))

# 负盈利的交易日平均利润
losing_trades = profits[profits < 0]
print('亏损交易数量：', len(losing_trades), round(100.0 * len(losing_trades)/len(c), 2), "%")
print('平均损失%：', round(np.mean(losing_trades) * 100, 2))

# map例题
def square(x):
    return x ** 2
print(map(square, [1,2,3,4,5]))    # 计算列表各个元素的平方,返回迭代器
print(list(map(square, [1,2,3,4,5])))  # 使用 list() 转换为列表
print(list(map(lambda x: x ** 2, [1, 2, 3, 4, 5])))   # 使用 lambda 匿名函数

# 4.10 动手实践：使用 hanning 函数平滑数据
import numpy as np
import matplotlib.pyplot as plt
import sys

N = int(sys.argv[1])
weights = np.hamming(N)  # hanning：对股票收益率数组进行了平滑处理。
print('权重：', weights)
bhp = np.loadtxt('BHP.csv', delimiter=',', usecols=(6,), unpack=True)
bhp_returns = np.diff(bhp) / bhp[ : -1]
smooth_bhp = np.convolve(weights/weights.sum(), bhp_returns)[N-1: -N+1]

vale = np.loadtxt('VALE.csv', delimiter=',', usecols=(6,), unack=True)
vale_returns = np.diff(vale) / vale[ : -1]
smooth_vale = np.convolve(weights/weights.sum(), vale_returns)[N-1: -N+1]

K = int(sys.argv[1])
t = np.arange(N - 1, len(bhp_returns))
poly_bhp = np.polyfit(t, smooth_bhp, K)
poly_vale = np.polyfit(t, smooth_vale, K)

poly_sub = np.polysub(poly_bhp, poly_vale)  # polysub：对两个多项式作差运算。
xpoints = np.roots(poly_sub)
print("交叉点：", xpoints)

reals = np.isreal(xpoints)  # isreal：判断数组元素是否为实数；
print("实数：", reals)

xpoints = np.select([reals], [xpoints])
# select：选出了实数元素；可以根据一组给定的条件，从一组元素中挑选出符合条件的元素并返回数组。
xpoints = xpoints.real
print("真正的交叉点：", xpoints)
print("Sans 0s：", np.trim_zeros(xpoints))  # trim_zeros：去掉数组首尾的0元素。

plt.plot(t, bhp_returns[N-1:], lw=1.0)
plt.plot(t, smooth_bhp, lw=2.0)
plt.plot(t, vale_returns[N-1:], lw=1.0)
plt.plot(t, smooth_vale, lw=2.0)
plt.show()


# hamming、blackman、bartlett以及kaiser简单使用。
import numpy as np
import matplotlib.pyplot as plt

window = np.hamming(142)
plt.plot(window)
plt.show()
# hamming：汉明窗是一个加权的余弦函数，hamming函数唯一的参数是输出点的个数。

from matplotlib.dates import datestr2num
closes = np.loadtxt('AAPL.csv', delimiter=',', usecols=(6,), converters={1:datestr2num}, unpack=True)
N = 10
window = np.blackman(N)
smoothed = np.convolve(window/window.sum(), closes, mode="same")
plt.plot(smoothed[N:-N], lw=2, label="smoothed")
plt.plot(closes[N:-N], label="closes")
plt.legend(loc="best")
plt.show()
# blackman：布莱克曼窗形式上是三项余弦值的加和。

window = np.bartlett(42)
plt.plot(window)
plt.show()
# bartlett：巴特利特窗是一种三角形平滑窗。

window = np.kaiser(42, 14)
plt.plot(window)
plt.show()
# kaiser：凯泽窗是以贝塞尔函数定义的；
# 第一个参数为输出点的个数, 第二个参数为贝塞尔函数中的参数值。