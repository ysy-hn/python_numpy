# 例题
# 9.2 动手实践：绘制多项式函数
import numpy as np
import matplotlib.pyplot as plt

func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
x = np.linspace(-10, 10, 30)
y = func(x)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使xlabel和ylabel可以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.plot(x, y)
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.show()

# 9.4 动手实践：绘制多项式函数及其导函数
import numpy as np
import matplotlib.pyplot as plt

func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
func1 = func.deriv(m=1)  # 1阶导数
x = np.linspace(-10, 10, 30)
y1 = func(x)
y2 = func1(x)

plt.plot(x, y1, 'ro', x, y2, 'g--')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.show()

# 9.5 子图
import numpy as np
import matplotlib.pyplot as plt

func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
x = np.linspace(-10, 10, 30)
y1 = func(x)
func1 = func.deriv(m=1)  # 一阶导数
y2 = func1(x)
func2 = func.deriv(m=2)  # 二阶导数
y3 = func2(x)

plt.subplot(312)
plt.plot(x, y1, 'r-')
plt.title('原函数')

plt.subplot(311)
plt.plot(x, y2, 'b^')
plt.title('一阶导数')

plt.subplot(313)
plt.plot(x, y3, 'go')
plt.title('二阶导数')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.show()

# 9.7 财经
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = ['sans-serif']  #设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

dates = ['2018/01/01','2018/1/2', '2018/1/03', '2018/01/4','2018/01/5','2018/01/6','2018/01/07','2018/01/08']  # 生成横纵坐标信息
date = [datetime.strptime(d, '%Y/%m/%d').date() for d in dates]
y = [25,18,13,14,12,17,16,15]

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))  # 配置横坐标
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(date[::2])  # 设置每隔多少距离一个刻度
plt.ylabel(u"y值")
plt.xlabel(u"时间(天)")

plt.plot(date, y, label=u"曲线")
plt.legend()
plt.gcf().autofmt_xdate()  # 自动旋转日期标记
plt.show()

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

# 9.9 直方图，hist函数可以绘制直方图；该函数的参数中有这样两项——包含数据的数组以及柱形的数量。
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.family'] = ['sans-serif']  #设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

dates = ['2018/01/01','2018/1/2', '2018/1/03', '2018/01/4','2018/01/5','2018/01/6','2018/01/07','2018/01/08']  # 生成横纵坐标信息
date = [datetime.strptime(d, '%Y/%m/%d').date() for d in dates]
filename = '阿里股票价格变化.csv'
closes = np.loadtxt(filename, delimiter=',', usecols=(1,), unpack=True)

fig = plt.figure()
plt.bar(date, closes)
fig.autofmt_xdate()
plt.ylabel("y值")
plt.xlabel("时间(天)")
plt.grid(True)
plt.show()

# 9.11 对数坐标图，当数据的变化范围很大时，对数坐标图（logarithmic plot）很有用。
#
#
# 9.13 散点图，用于绘制同一数据集中的两种数值变量。
#
#
# 9.15 着色
#
#
# 9.17 图例和注释


# 9.19 三维绘图，Axes3D对象
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 建议使用该形式
# ax = Axes3D(fig)  # 和上面功能一致

u = np.linspace(-1, 1, 100)
x, y = np.meshgrid(u, u)  # meshgrid:创建二维坐标网格
z = x ** 2 + y ** 2
ax.plot_surface(x, y, z, rstride=4, cstride=4, cmap=cm.YlGnBu_r)
# plot_surface：创建3d画图时使用。
# 指定行和列的步幅，以及绘制曲面所用的色彩表（color map）
plt.show()

# 9.21 等高线图，填充的和非填充的。
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111)

u = np.linspace(-1, 1, 100)

x, y = np.meshgrid(u, u)
z = x ** 2 + y ** 2
# ax.contour(x, y, z)  # 未填充的等高线
ax.contourf(x, y, z)  # 填充的等高线

plt.show()

# 9.23 动画，需要定义一个回调函数，用于定期更新屏幕上的内容。我们还需要一个函数来生成图中的数据点。
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot(111)
N = 10
x = np.random.rand(N)  # rand:返回10个0-1之间的随机数
y = np.random.rand(N)
z = np.random.rand(N)
circles, triangles, dots = ax.plot(x, 'ro', y, 'g^', z, 'b.')
ax.set_ylim(0, 1)
plt.axis('off')

def update(data):
    circles.set_ydata(data[0])
    triangles.set_ydata(data[1])
    return circles, triangles

def generated():
    while True:
        yield np.random.rand(2, N)


anim = animation.FuncAnimation(fig, update, generated, interval=150)
plt.show()

# 1、plt.axis(‘square’)，作图为正方形，并且x,y轴范围相同。
# 2、plt.axis(‘equal’)，x,y轴刻度等长。
# 3、plt.axis(‘off’)，关闭坐标轴。
# 4、plt.axis([a, b, c, d])，设置x轴的范围为[a, b]，y轴的范围为[c, d]。
# FuncAnimation:展示动画时使用。

























