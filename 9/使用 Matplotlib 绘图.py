# 知识
# 首先确保已经安装python，然后用pip来安装matplotlib模块。（推荐学习：Python视频教程）
# 进入到cmd窗口下，执行python -m pip install -U pip setuptools进行升级。
# 接着键入python -m pip install matplotlib进行自动的安装，系统会自动下载安装包。
# 安装完成后，可以用python -m pip list查看本机的安装的所有模块，确保matplotlib已经安装成功。

# 9.2 动手实践：绘制多项式函数
# poly1d函数：创建多项式。

# 9.4 动手实践：绘制多项式函数及其导函数
# .deriv函数：求导；func1 = func.deriv(m=1)。

# 9.5 子图
# subplot函数：行、列、子图序号。如311，子图将被组织成3行1列第一个子图。

# 9.7 财经
# matplotlib.finance包中的函数可以从雅虎财经频道（http://finance.yahoo.com/）下载股价数据，
# 并绘制成K线图（candlestick）。
# 创建所谓的定位器（locator），这些来自matplotlib.dates包中的对象可以在x轴上定位月份和日期。
# 创建一个日期格式化器（date formatter）以格式化x轴上的日期。该格式化器将创建一个字符串，
# 包含简写的月份和年份。
# figure对象——这是绘图组件的顶层容器。
# autofmt_xdate()：旋转标签。

# 9.9 直方图，hist函数可以绘制直方图；该函数的参数中有这样两项——包含数据的数组以及柱形的数量。

# 9.11 对数坐标图，当数据的变化范围很大时，对数坐标图（logarithmic plot）很有用。
# semilogx函数：对x轴取对数；
# semilogy函数：对y轴取对数；
# loglog函数：同时对x轴和y轴取对数。

# 9.13 散点图，用于绘制同一数据集中的两种数值变量。
# 数据点的颜色与股票收益率相关联，数据点的大小与成交量的变化相关联。c（颜色）= ，s（点大小）= 。
# grid(True) ：网格线。

# 9.15 着色
# fill_between函数：使用指定的颜色填充图像中的区域。
# alpha通道的取值，该函数的where参数可以指定着色的条件。

# 9.17 图例和注释
# legend函数：创建透明的图例，并由Matplotlib自动确定其摆放位置。
# annotate函数：在图像上精确地添加注释，并有很多可选的注释和箭头风格。

# 9.19 三维绘图，Axes3D对象
# meshgrid函数：创建一个二维的坐标网格。
# plot_surface(x, y, z, rstride（行）=4, cstride（列）=4, cmap（曲面）=cm.YlGnBu_r)
# 行和列的步幅，以及绘制曲面所用的色彩表（color map）

# 9.21 等高线图，填充的和非填充的。
# contour函数：创建一般的等高线图。
# contourf函数：创建色彩填充的等高线图。

# 9.23 动画，需要定义一个回调函数，用于定期更新屏幕上的内容。我们还需要一个函数来生成图中的数据点。

# np.random.rand(N)  # rand:返回10个0-1之间的随机数
# np.random.randn(d0,d1,d2……dn)
# 1)当函数括号内没有参数时，则返回一个浮点数；
# 2）当函数括号内有一个参数时，则返回秩为1的数组，不能表示向量和矩阵；
# 3）当函数括号内有两个及以上参数时，则返回对应维度的数组，能表示向量或矩阵；
# 4）np.random.standard_normal（）函数与np.random.randn()类似，
# 但是np.random.standard_normal（）的输入参数为元组（tuple）.
# 5)np.random.randn()的输入通常为整数，但是如果为浮点数，则会自动直接截断转换为整数

# 1、plt.axis(‘square’)，作图为正方形，并且x,y轴范围相同。
# 2、plt.axis(‘equal’)，x,y轴刻度等长。
# 3、plt.axis(‘off’)，关闭坐标轴。
# 4、plt.axis([a, b, c, d])，设置x轴的范围为[a, b]，y轴的范围为[c, d]。
# FuncAnimation:展示动画时使用。








