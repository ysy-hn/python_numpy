# 知识
# 7.1 排序
# 1、sort函数返回排序后的数组；
# 2、lexsort函数根据键值的字典序进行排序；
# 3、argsort函数返回输入数组排序后的下标；
# 4、ndarray类的sort方法可对数组进行原地排序；
# 5、msort函数沿着第一个轴排序；
# 6、sort_complex函数对复数按照先实部后虚部的顺序进行排序。

# 7.2 动手实践：按字典序排序
# lexsort函数：返回输入数组按字典序排序后的下标。

# 7.3 复数，NumPy中有专门的复数类型，使用两个浮点数来表示复数。
# seed()函数：改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数。x，改变随机数生成器的种子 seed。
# sort_complex函数：对复数按照先实部后虚部的顺序进行排序。

# 7.5 搜索
# argmax函数：返回数组中最大值对应的下标。
# nanargmax函数：提供相同的功能，但忽略NaN值；np.array([np.nan, 2, 4])
# argmin和nanargmin函数：功能类似，换成了最小值。
# argwhere函数根据条件搜索非零的元素，并分组返回对应的下标。
# searchsorted函数：可以为指定的插入值寻找维持数组排序的索引位置。
# extract函数：返回满足指定条件的数组元素。

# 7.6 动手实践：使用 searchsorted 函数
# searchsorted函数：可以为指定的插入值寻找维持数组排序的索引位置；
# 简单的说，返回将随机数放置到有序的数组中的合适索引位置。
# insert函数：构建完整的数组； np.insert(数组, 索引位置, [-2, 7])

# 7.7 数组元素抽取
# extract函数：可以根据某个条件从数组中抽取元素。
# nonzero函数：专门用来抽取非零的数组元素。

# 7.9 金融函数
# 1、fv函数计算所谓的终值（future value），即基于一些假设给出的某个金融资产在未来某一时间点的价值。
# 2、pv函数计算现值（present value），即金融资产当前的价值。
# 3、npv函数返回的是净现值（net present value），即按折现率计算的净现金流之和。
# 4、pmt函数根据本金和利率计算每期需支付的金额。
# 5、irr函数计算内部收益率（internal rate of return）。内部收益率是是净现值为0时的有效利率，不考虑通胀因素。
# 6、mirr函数计算修正后内部收益率（modified internal rate of return），是内部收益率的改进版本。
# 7、nper函数计算定期付款的期数。
# 8、rate函数计算利率（rate of interest）。

# 7.10 动手实践：计算终值，终值是基于一些假设给出的某个金融资产在未来某一时间点的价值；
# 终值决定于4个参数——利率、期数、每期支付金额以及现值。
# fv函数：计算所谓的终值。

# 7.11 现值，现值是指资产在当前时刻的价值。
# pv函数：计算现值。
# 该函数和fv函数是镜像对称的，同样需要利率、期数、每期支付金额这些参数，不过这里输入为终值，输出为现值。

# 7.13 净现值，净现值定义为按折现率计算的净现金流之和。
# npv函数：返回净现值。需要两个参数，即利率和一个表示现金流的数组。

# 7.15 内部收益率，内部收益率是净现值为0时的有效利率，不考虑通胀因素。
# irr函数：根据给定的现金流数据返回对应的内部收益率。

# 7.17 分期付款
# pmt函数可以根据利率和期数计算贷款每期所需支付的资金。

# 7.19 付款期数
# nper函数可以计算分期付款所需的期数。所需的参数为贷款利率、固定的月供以及贷款额。

# 7.21 利率
# rate函数根据给定的付款期数、每期付款资金、现值和终值计算利率。

# 7.23 窗函数，是信号处理领域常用的数学函数，相关应用包括谱分析和滤波器设计等。

# 7.24 动手实践：绘制巴特利特窗
# bartlett：巴特利特窗是一种三角形平滑窗。

# 7.25 布莱克曼窗，是三项余弦值的加和。
# blackman：该函数唯一的参数为输出点的数量，如果数量为0或小于0，则返回一个空数组。

# 7.26 动手实践：使用布莱克曼窗平滑股价数据
# convolve函数：卷积函数。

# 7.27 汉明窗，是一个加权的余弦函数
# hamming：hamming函数唯一的参数是输出点的个数。如果数量为0或小于0，则返回一个空数组。

# 7.29 凯泽窗，是以贝塞尔函数（Bessel function）定义的。
# kaiser：第一个参数为输出点的个数, 第二个参数为贝塞尔函数中的参数值。
# 如果数量为0或小于0，则返回一个空数组。

# 7.31 专用数学函数，贝塞尔函数是贝塞尔微分方程的标准解函数。以i0表示第一类修正的零阶贝塞尔函数。

# 7.33 sinc 函数，sinc函数在数学和信号处理领域被广泛应用。
# sinc函数：在NumPy中有同名函数sinc，并且该函数也有一个二维版本。sinc是一个三角函数。
# sinc函数的定义是：sinc(a) =sin(πa)/(πa)，我们常看到模糊的叙述：“sinc函数是sin(x)/x的一般形式。 ”
# sinc是衰减到振幅为1/x的正弦波。

# numpy.outer(a, b, out=None)：计算两个向量的外积。
# a: [数组]，第一个输入向量。如果输入不是一维的，则将其展平。
# b: [数组]，第二个输入向量。如果输入不是一维的，则将其展平。
# out: [ndarray，可选]存储结果的位置。
# Return: 返回两个向量的外积。 out[i，j] = a[i] * b[j]
# 例如：
import numpy as np

a = np.ones(4)  # ones：初始化数组为1；zeros：初始化数组为0.
b = np.linspace(-1, 2, 4)
print(b)
gfg = np.outer(a, b)
print(gfg)

# 注意：金融函数无法使用，暂不知原因。



# 例题
# 例题
# 7.2 动手实践：按字典序排序
import numpy as np
import datetime
def datestr2num(s):
    return datetime.datetime.strptime(s.decode('ascii'), "%m/%d/%Y").toordinal()
filename = '阿里股票价格变化.csv'
dates, closes = np.loadtxt(filename, delimiter=',', usecols=(0, 1),
                          converters={0: datestr2num}, unpack=True)
indices = np.lexsort((dates, closes))
print("indices:", indices)
print(["%s %s" % (datetime.date.fromordinal(int(dates[i])), closes[i]) for i in indices])

# 7.3 复数
import numpy as np
np.random.seed()
complex_numbers = np.random.random(5) + 1j * np.random.random(5)
print('complex number(杂乱数字）：', complex_numbers)
print('排序后的数字：', np.sort_complex(complex_numbers))

# 7.6 动手实践：使用 searchsorted 函数
import numpy as np
a = np.arange(5)
indices = np.searchsorted(a, [-2, 7, 8, 1.5])
print('打印出合适索引位置：', indices)
print('填充到有序数组中：', np.insert(a, indices, [-2, 7, 8, 1.5]))

# 7.7 数组元素抽取
import numpy as np
a = np.arange(7)
condition = (a % 2) == 0
print('even numbers(偶数)：', np.extract(condition, a))
print('non zero(非0数）：', np.nonzero(a))

# 7.10 动手实践：计算终值
import numpy as np
from matplotlib.pyplot import plot, show
print("Future value", np.fv(0.03/4, 5 * 4, -10, -1000))
fvals = []
for i in range(1, 10):
    fvals.append(np.fv(.03/4, i * 4, -10, -1000))
plot(fvals, 'bo')
show()

# 7.11 现值
import numpy as np
print(np.pv(0.03/4, 5 * 4, -10, 1376.09633204))

# 7.13 净现值
import numpy as np
cashflows = np.random.randint(100, size=5)
cashflows = np.insert(cashflows, 0, -100)
print("Cashflows", cashflows)
print("Net present value", np.npv(0.03, cashflows))

# 7.15 内部收益率
import numpy as np
print("Internal rate of return", np.irr([-100, 38, 48, 90, 17, 36]))

# 7.17 分期付款
import numpy as np
print('分期付款：', np.pmt(0.10/12, 12*30, 1000000))

# 7.19 付款期数
import numpy as np
print('付款期数：', np.nper(0.10/12, -100, 9000))

# 7.21 利率
import numpy as np
print('利率：', 12 * np.rate(167, -100, 9000, 0))

# 7.24 动手实践：绘制巴特利特窗
import numpy as np
import matplotlib.pyplot as plt
window = np.bartlett(10)
plt.plot(window)
plt.show()

# 7.26 动手实践：使用布莱克曼窗平滑股价数据
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import datestr2num

closes = np.loadtxt('阿里股票价格变化.csv', delimiter=',', usecols=(1,), converters={0: datestr2num}, unpack=True)
N = int(input('请输入数值：'))
window = np.blackman(N)
smoothed = np.convolve(window/window.sum(), closes, mode='same')
plt.plot(smoothed[N:-N], lw=2, label='smoothed')
plt.plot(closes[N:-N], label='closes')
plt.legend(loc='best')
plt.show()

# 7.27 汉明窗，是一个加权的余弦函数
import numpy as np
import matplotlib.pyplot as plt

window = np.hamming(100)
plt.plot(window)
plt.show()

# 7.29 凯泽窗，是以贝塞尔函数（Bessel function）定义的。
import numpy as np
import matplotlib.pyplot as plt

window = np.kaiser(100, 10)
plt.plot(window)
plt.show()

# 7.31 专用数学函数
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 100)
vals = np.i0(x)
plt.plot(x, vals)
plt.show()

# 7.33 sinc 函数
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4, 100)
# vals = np.sinc(x)
# plt.plot(x, vals)
# plt.show()

xx = np.outer(x, x)  # outer函数,计算两个向量的外积。
vals = np.sinc(xx)

plt.imshow(vals)
plt.show()
