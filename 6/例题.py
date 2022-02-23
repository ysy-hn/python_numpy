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