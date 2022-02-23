# 知识
# 6.1 线性代数numpy.linalg模块包含线性代数的函数。
# 使用这个模块，可以计算逆矩阵、求特征值、解线性方程组以及求解行列式等。
#
# 6.2 动手实践：计算逆矩阵
# mat函数：创建矩阵。
# inv函数：计算逆矩阵;只接受方阵作为输入矩阵。
#
# 6.3 求解线性方程组，Ax = b，A 为矩阵，b 为一维或二维的数组，x 是未知变量。
# solve函数：求解线性方程组。
#
# 6.5 特征值和特征向量,特征值即方程Ax=ax的根，是一个标量。A是一个二维矩阵，x是一个一维向量。
# eigvals函数:求解特征值.
# eig函数:求解特征值和特征向量;将返回一个元组，按列排放着特征值和特征向量，第一列为特征值，第二列为特征向量。
#
# 6.7 奇异值分解,SVD是一种因子分解运算，将一个矩阵分解为3个矩阵的乘积。
# 函数返回3个矩阵——U、Sigma和V，其中U和V是正交矩阵，Sigma包含输入矩阵的奇异值。
# svd函数:分解矩阵;返回等式中左右两端的两个正交矩阵U和V，以及中间的奇异值矩阵Sigma。
# diag函数:生成完整的奇异值矩阵.
#
# 6.9 广义逆矩阵,使用numpy.linalg模块中的pinv函数进行求解.
# inv函数：计算逆矩阵;只接受方阵作为输入矩阵。
# pinv函数:计算广义逆矩阵；没有这个限制。
#
# 6.11 行列式,行列式的值为正表示保持了空间的定向（顺时针或逆时针），为负则表示颠倒了空间的定向。
# det函数:计算行列式.
#
# 6.13 快速傅里叶变换,FFT(快速傅里叶变换)是一种高效的计算DFT（离散傅里叶变换）的算法。
# DFT在信号处理、图像处理、求解偏微分方程等方面都有应用.
# 在NumPy中，有一个名为fft的模块提供了快速傅里叶变换的功能。在这个模块中，许多函数都是成对存在的，
# 也就是说许多函数存在对应的逆操作函数。例如，fft和ifft函数就是其中的一对.
# fft函数:对余弦波信号进行傅里叶变换。
# ifft函数：fft函数的逆操作函数；对变换后的结果应用ifft函数，可以近似地还原初始信号。
#
# all(iterable) 函数:判断给定的可迭代参数iterable中的所有元素是否都为TRUE，如果是返回True，否则返回 False。
# 元素除了是0、空、None、False外都算 True。
#
# 6.15 移频
# numpy.linalg模块中的fftshift函数可以将FFT输出中的直流分量移动到频谱的中央。
# ifftshift函数则是其逆操作。
# fftshift函数:进行移频操作。
# ifftshift函数:进行逆操作，将还原移频操作前的信号。
#
# 6.17 随机数,真随机数的产生很困难，因此在实际应用中我们通常使用伪随机数
# 二项分布是n个独立重复的是/非试验中成功次数的离散概率分布，这些概率是固定不变的，与试验结果无关
# random模块中的binomial(n,p,size=None):n,一次试验的样本数n，并且相互不干扰;
# p,事件发生的概率p，范围[0,1];size,限定了返回值的形式（具体见上面return的解释）和实验次数;
# return返回值：以size给定的形式，返回每次试验事件发生的次数，次数大于等于0且小于等于参数n。
#
# 6.19 超几何分布,是一种离散概率分布，它描述的是一个罐子里有两种物件，
# 无放回地从中抽取指定数量的物件后，抽出指定种类物件的数量。
# NumPy random模块中的hypergeometric函数可以模拟这种分布.
# hypergeometric(好选择数，坏选择数，抽样数，输出形状）；
# 例如： np.random.hypergeometric(25, 1, 3, size=len(points))
#
# 6.21 连续分布,正态分布，连续分布可以用PDF（概率密度函数）来描述。
# 随机变量落在某一区间内的概率等于概率密度函数在该区间的曲线下方的面积。
# NumPy random模块中的normal函数:产生指定数量的随机数。
#
# 6.23 对数正态分布,对数正态分布是自然对数服从正态分布的任意随机变量的概率分布。
# NumPy random模块中的lognormal函数模拟了这个分布。
# np.random.lognormal函数：模拟对数正态分布。

# 直方图简介：
# plt.hist()：直方图，一种特殊的柱状图。
# 将统计值的范围分段，即将整个值的范围分成一系列间隔，然后计算每个间隔中有多少值。
# 直方图也可以被归一化以显示“相对”频率。 然后，它显示了属于几个类别中的每个类别的占比，其高度总和等于1。
#
# 常用参数解释：
# x: 作直方图所要用的数据，必须是一维数组；多维数组可以先进行扁平化再作图；必选参数；
# bins: 直方图的柱数，即要分的组数，默认为10；
# range：元组(tuple)或None；剔除较大和较小的离群值，给出全局范围；如果为None，则默认为(x.min(), x.max())；即x轴的范围；
# density：布尔值。如果为true，则返回的元组的第一个参数n将为频率而非默认的频数；
# color：具体颜色，数组（元素为颜色）或None。
# facecolor：颜色；
# edgecolor: 直方图边框颜色；
# alpha: 透明度；
# bottom：数组，标量值或None；每个柱子底部相对于y=0的位置。如果是标量值，则每个柱子相对于y=0向上/
# 向下的偏移量相同。如果是数组，则根据数组元素取值移动对应的柱子；即直方图上下便宜距离；
# align：{‘left’, ‘mid’, ‘right’}；‘left’：柱子的中心位于bins的左边缘；
# ‘mid’：柱子位于bins左右边缘之间；‘right’：柱子的中心位于bins的右边缘；
# log：布尔值。如果取值为True，则坐标轴的刻度为对数刻度；
# 如果log为True且x是一维数组，则计数为0的取值将被剔除，仅返回非空的(frequency, bins, patches）；
# label：字符串（序列）或None；有多个数据集时，用label参数做标注区分；
# stacked：布尔值。如果取值为True，则输出的图为多个数据集堆叠累计的结果；
# 如果取值为False且histtype=‘bar’或’step’，则多个数据集的柱子并排排列；
#
# weights：与x形状相同的权重数组；将x中的每个元素乘以对应权重值再计数；如果normed或density取值为True，则会对权重进行归一化处理。这个参数可用于绘制已合并的数据的直方图；
# cumulative：布尔值；如果为True，则计算累计频数；如果normed或density取值为True，则计算累计频率；
# orientation：{‘horizontal’, ‘vertical’}：如果取值为horizontal，则条形图将以y轴为基线，水平排列；简单理解为类似bar()转换成barh()，旋转90°；
# rwidth：标量值或None。柱子的宽度占bins宽的比例；
# normed（不推荐使用，建议改用density参数）：是否将得到的直方图向量归一化，即显示占比，默认为0，不归一化；
# histtype：{‘bar’, ‘barstacked’, ‘step’, ‘stepfilled’}；'bar’是传统的条形直方图；
# 'barstacked’是堆叠的条形直方图；'step’是未填充的条形直方图，只有外边框；
# ‘stepfilled’是有填充的直方图；当histtype取值为’step’或’stepfilled’，rwidth设置失效，
# 即不能指定柱子之间的间隔，默认连接在一起；
#
# 返回值（用参数接收返回值，便于设置数据标签）：
# n：直方图向量，即每个分组下的统计值，是否归一化由参数normed设定。当normed取默认值时，n即为直方图各组内元素的数量（各组频数）；
# bins: 返回各个bin的区间范围；
# patches：返回每个bin里面包含的数据，是一个list。
# 其他参数与plt.bar()类似。


# 例题
# 6.2 动手实践：计算逆矩阵
import numpy as np

A = np.mat("0 1 2; 1 0 3; 4 -3 8")
print(A)
inverse = np.linalg.inv(A)
print(inverse)
print(inverse * A)

# 6.3 求解线性方程组
import numpy as np

A = np.mat("1 -2 1;0 2 -8;-4 5 9")
b = np.array([0, 8, -9])
print('A:', A)
print('b:', b)
x = np.linalg.solve(A, b)
print('Solution:', x)
print('check:', np.dot(A, x))

# 6.5 特征值和特征向量
import numpy as np

A = np.mat("3 -2;1 0")
print('A:', A)
print('Eigenvalues(特征值）:', np.linalg.eigvals(A))
eigenvalues, eigenvectors = np.linalg.eig(A)
print('特征值：', eigenvalues)
print('特征向量：', eigenvectors)
for i in range(len(eigenvalues)):
    print("Left", np.dot(A, eigenvectors[:,i]))
    print("Right", eigenvalues[i] * eigenvectors[:,i])

# 6.7 奇异值分解
import numpy as np

A = np.mat("4 11 14;8 7 -2")
u, sigma, v = np.linalg.svd(A, full_matrices=False)
print('u:', u)
print('sigma:', sigma)
print('v:', v)
print('product(产品）：', u * np.diag(sigma) * v)

# 6.9 广义逆矩阵
import numpy as np
A = np.mat("4 11 14;8 7 -2")
pseudoinv = np.linalg.pinv(A)
print('Pseudo inverse(广义逆矩阵）：', pseudoinv)
print('check(检查）：', A * pseudoinv)  # 原矩阵与广义逆矩阵相乘，得到的结果为一个近似单位矩阵。

# 6.11 行列式
import numpy as np
A = np.mat("3 4;5 6")
print('Determinant:', np.linalg.det(A))

# 6.13 快速傅里叶变换
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)
transformed = np.fft.fft(wave)
print(x)
print(np.all(np.abs(np.fft.ifft(transformed) - wave) < 10 ** -9))  # ifft:近似地还原初始信号
# all(iterable) 函数:判断给定的可迭代参数iterable中的所有元素是否都为TRUE，如果是返回True，否则返回 False。
# 元素除了是0、空、None、False外都算 True。
plt.plot(transformed)
plt.show()

# 6.15 移频
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 30)
wave = np.cos(x)
transformed = np.fft.fft(wave)
shifted = np.fft.fftshift(transformed)
print(np.all(np.abs(np.fft.ifftshift(shifted) - transformed) < 10 ** -9))
plt.plot(transformed, lw=2)
plt.plot(shifted, lw=3)
plt.show()

# 6.17 随机数
import numpy as np
import matplotlib.pyplot as plt

cash = np.zeros(10000)
cash[0] = 1000
outcome = np.random.binomial(9, 0.5, size=len(cash))
for i in range(1, len(cash)):
    if outcome[i] < 5:
        cash[i] = cash[i-1] - 1
    elif outcome[i] < 10:
        cash[i] = cash[i-1] + 1
    else:
        raise AssertionError('Unexpected outcome（意外结果）：', outcome)
print('最小值：{};最大值：{}。'.format(outcome.min(), outcome.max()))
plt.plot(np.arange(len(cash)), cash)
plt.show()

# 6.19 超几何分布
import numpy as np
import matplotlib.pyplot as plt

points = np.zeros(100)
outcomes = np.random.hypergeometric(25, 1, 3,size=len(points))

for i in range(len(points)):
    if outcomes[i] == 3:
        points[i] = points[i-1] + 1
    elif outcomes[i] == 2:
        points[i] = points[i - 1] - 6
    else:
        print(outcomes[i])
plt.plot(np.arange(len(points)), points)
plt.show()

# 6.21 连续分布,正态分布
import numpy as np
import matplotlib.pyplot as plt

N = 10000
normal_values = np.random.normal(size=N)
dummy, bins, dummy = plt.hist(normal_values, int(np.sqrt(N)), density=True, facecolor="blue", edgecolor="black", alpha=0.5)
# 绘制分布直方图和理论上的概率密度函数（均值为0、方差为1的正态分布）曲线。
# 指定y轴数据源, 指定显示的小长条个数.
# density：密度布尔值。如果为true，则返回的元组的第一个参数n将为频率而非默认的频数。
# facecolor：颜色；edgecolor：边缘颜色；alpha:透明度。
sigma = 1
mu = 0
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), lw=2)
# 第一个参数是代表x轴的数组, 第二个参数是代表y轴的数组
# 本例中第二个参数是一个定义域为bins的映射
plt.show()

# 6.23 对数正态分布
import numpy as np
import matplotlib.pyplot as plt
N = 10000
lognormal_values = np.random.lognormal(size=N)
dummy, bins, dummy = plt.hist(lognormal_values, int(np.sqrt(N)), density=True, lw=1)
sigma = 1
mu = 0
x = np.linspace(min(bins), max(bins), len(bins))
pdf = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi))
plt.plot(x, pdf, lw=3)
plt.show()


