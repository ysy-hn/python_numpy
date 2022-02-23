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
