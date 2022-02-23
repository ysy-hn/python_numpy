# 知识
# 5.2 动手实践：创建矩阵
# mat函数：创建矩阵；形如：np.mat('1 2 3; 4 5 6; 7 8 9').
# 使用NumPy数组进行创建矩阵：np.mat(np.arange(9).reshape(3, 3))
# T属性:获取转置矩阵.
# I属性获取逆矩阵，I属性只存在于matrix对象上。
#
# I属性只存在于matrix对象上，而不是ndarrays。
# 使用numpy.linalg.inv反转数组（但有些矩阵无法求逆)。
# np.linalg.pinv函数:广义逆矩阵.
# np.linalg.det函数:计算矩阵的行列式.
# SVD:是一种因子分解运算, 将一个矩阵分解为3个矩阵的乘积;其中, 奇异值矩阵是对角线矩阵.
# np.linalg.svd函数, 可以对矩阵进行奇异值分解.
# U: 正交矩阵;sigma: 表示奇异值矩阵对角线的数组, 其他非对角线元素均为0.
# V: 正交矩阵;np.diag函数, 得出完整的奇异值矩阵.
#
# 5.4 动手实践：从已有矩阵创建新矩阵
# bmat函数：分块矩阵，多个较小矩阵复合成大矩阵。
#
# 5.6 动手实践：创建通用函数
# zeros_like(a):创建和a形状相同但元素全为0的数组；
# flat属性为我们提供了一个扁平迭代器，可以逐个设置数组元素的值；flat属性是一个可赋值的属性。
# frompyfunc(函数, 输入参数个数, 输出参数个数) ：创建通用函数；
#
# 5.8 动手实践：在 add 上调用通用函数的方法
# add.reduce函数：对数组所有元素求和。
# add.accumulate函数：存储运算的中间结果并返回；等价于直接调用cumsum函数;简单的说就是对元素求和并返回值。
# add.reduceat函数：输入一个数组以及一个索引值列表作为参数；以索引值列表为参数进行reduce操作。
# add.outer函数：返回一个数组，它的秩（rank）等于两个输入数组的秩的和。
# 简单的说，前一个参数的数组每个元素与后一个参数数组每个元素相加，最后维度有前一个参数数组元素个数决定。
#
# matlab中cumsum函数通常用于计算一个数组各行的累加值;
# 格式一：B = cumsum(A),这种用法返回数组不同维数的累加和。
# 如果A是一个矩阵/向量，cumsum(A)返回一个和A同行同列的矩阵/向量，所有元素累加和；
# 如果A是一个多维数组， cumsum(A)只对A中第一个非奇异维进行计算。
# 格式二：B = cumsum(A,n)返回A中由标量n所指定的维数的累加和。
# 例如：cumsum(A,1)返回的是沿着第一维（各列）的累加和，cumsum(A,2)返回的是沿着第二维（各行）的累加和。
#
# 5.9 算术运算：基本算术运算符+、-和*隐式关联着通用函数add、subtract和multiply。
#
# 5.10 动手实践：数组的除法运算
# divide函数、true_divide函数、运算符/，三者相等：与数学中的除法定义更为接近，即返回除法的浮点数结果而不作截断。
# floor_divide函数（等于运算符//）：返回整数结果，相当于先调用divide函数再调用floor函数；
# 对浮点数进行向下取整并返回整数。
#
# 5.12 动手实践：模运算
# %、remainder函数、 mod函数：逐个返回两个数组中元素相除后的余数；如果第二个数字为0，则直接返回0。
# fmod函数：所得余数的正负由被除数决定，与除数的正负无关，其它与上述函数一致。

# 求摸和求余简介：
# 当a和b正负号一致时，求模运算和求余运算所得的c的值一致，因此结果一致。当正负号不一致时，结果不一样。
# 对于整型数a，b来说，取模运算或者求余运算的方法都是：
# 1.求整数商： c = [a/b];
# 2.计算模或者余数： r = a - c*b.
# 求模运算和求余运算在第一步不同: 取余运算在取c的值时，向0方向舍入(fix()函数)；
# 而取模运算在计算c的值时，向负无穷方向舍入(floor()函数)。
# 例1.计算：-7 Mod 4
# 第一步：求整数商c：
# ①进行求模运算c = [a/b] = -7 / 4 = -2（向负无穷方向舍入），
# ②进行求余运算c = [a/b] = -7 / 4 = -1（向0方向舍入）；
# 第二步：计算模和余数的公式相同，但因c的值不同，
# ①求模时：r = a - c*b =-7 - (-2)*4 = 1，
# ②求余时：r = a - c*b = -7 - (-1)*4 =-3。
# 例2.计算：7 Mod 4
# 那么：a = 7；b = 4
# 第一步：求整数商c：
# ①进行求模运算c = [a/b] = 7 / 4 = 1
# ②进行求余运算c = [a/b] = 7 / 4 = 1
# 第二步：计算模和余数的公式相同
# ①求模时：r = a - c*b =7 - (1)*4 = 3，
# ②求余时：r = a - c*b = 7 - (1)*4 =3。
# 另外各个环境下%运算符的含义不同，比如c/c++，java 为取余，而python则为取模。
# 补充：
# 7 mod 4 = 3（商 = 1 或 2，1<2，取商=1）
# -7 mod 4 = 1（商 = -1 或 -2，-2<-1，取商=-2）
# 7 mod -4 = -1（商 = -1或-2，-2<-1，取商=-2）
# -7 mod -4 = -3（商 = 1或2，1<2，取商=1）

# 5.14 动手实践：计算斐波那契数列,后一个数是前两个数的和。
# matrix函数:创建矩阵。
# rint函数:对浮点数取整但不改变浮点数类型。
#
# 5.15 利萨茹曲线，x = A sin(at + n/2) ，y = B sin(bt)。见下面例题。
#
# 5.17 方波,方波可以近似表示为多个正弦波的叠加。事实上，任意一个方波信号都可以用无穷傅里叶级数来表示。
# 傅里叶级数（Fourier series）是以正弦函数和余弦函数为基函数的无穷级数。见下面例题。
#
# 5.19 锯齿波和三角波，锯齿波和三角波也是常见的波形。我们也可以将它们表示成无穷傅里叶级数。见下面例题。
#
# 5.21 位操作函数和比较函数,位操作函数可以在整数或整数数组的位上进行操作，它们都是通用函数。
# ^、&、|、<<、>>等位操作符在NumPy中也有对应的部分，<、>、==等比较运算符也是如此。有了这些操作符，
# 你可以在代码中玩一些高级技巧以提升代码的性能。但会使代码变得难以理解，需谨慎使用
#
# 5.22 动手实践：玩转二进制位
# 1、检查两个整数的符号是否一致，使用XOR或者^操作符；
# XOR操作符又被称为不等运算符，因此当两个操作数的符号不一致时，XOR操作的结果为负数。
# less函数:对应于< 操作符；
# bitwise_xor函数：对应于^操作符；按元素计算两个数组的按位XOR。

# 2、检查一个数是否为2的幂数，在2的幂数以及比它小1的数之间执行位与操作AND，返回得到0；
# equal函数：对应于==操作符；是否相等。
# bitwise_and函数对应于：&操作符；对数组中整数的二进制形式执行位与运算。

# 3、计算一个数被2的幂数整除后的余数（计算余数的技巧实际上只在模为2的幂数（如4、8、16等）时有效）；
# left_shift函数：对应于<<操作符；对数值进行左移位运算：二进制数值向左移位，右边补0。

# bitwise_and函数对应于：&操作符；对数组中整数的二进制形式执行位与运算。
# bitwise_or函数：对数组元素执行位或操作。
# bitwise_xor函数：对应于^操作符；按元素计算两个数组的按位XOR。
# invert函数：按元素计算按位求逆，或按位求非，按位取反。
# left_shift函数：对应于<<操作符；对数值进行左移位运算：二进制数值向左移位，右边补0。
# right_shift函数：向右移动二进制表示的位。

# numpy中矩阵和数组的区别
# 1、矩阵只能为2维的，而数组可以是任意维度的。
# 2、矩阵和数组在数学运算上会有不同的结构。
# 一、矩阵的创建：
# 1、采用mat函数创建矩阵：numpy.mat(data, dtype=None)；相当于numpy.matrix(data, copy=False)）。
# 2、采用matrix函数创建矩阵：numpy.matrix(data, dtype=None, copy=True)。
# 二、数组的创建
# 1、通过传入列表创建。
# 2、通过range()和reshape()创建。
# 3、linspace()和reshape()创建。
# 4、通过内置的一些函数创建。
# 三、矩阵和数组的数学运算
# 1、矩阵的乘法和加法：
# 矩阵的乘法、加法和线性代数的矩阵加法、乘法一致，运算符号也一样用*，**表示平方，例如e**2 =e*e。
# 2、数组的加法和乘法：
# 数组的乘法、加法为相应位置的数据乘法、加法。


# 例题
# 5.2 动手实践：创建矩阵
import numpy as np

A = np.mat('1 2 3; 4 5 6; 7 8 9')
print('创建矩阵1：', A)
B = np.mat(np.arange(9).reshape(3, 3))
print('创建矩阵2：', B)
C = np.arange(9).reshape(3, 3)
print('创建矩阵3：', C)
print('矩阵转置：', B.T, C.T)
print('行列式：', np.linalg.det(A))
print(np.linalg.pinv(A))

# 5.4 动手实践：从已有矩阵创建新矩阵
import numpy as np

A = np.eye(2)
B = 2 * A
C = np.bmat('A B; A B')
print('大矩阵：', C)

# 5.6 动手实践：创建通用函数
import numpy as np

def ultimate_answer(a):
    result = np.zeros_like(a)
    result.flat = 42
    return result

ufunc = np.frompyfunc(ultimate_answer, 1, 1)
print('结果1：', ufunc(np.arange(4)))
print('结果2：', ufunc(np.arange(4).reshape(2, 2)))

b = np.arange(4).reshape(2, 2)
print(b)
f = b.flat
print(f)
for item in f:
    print(item)
print(b.flat[2])
b.flat = 7
print(b)

# 5.8 动手实践：在 add 上调用通用函数的方法
import numpy as np

a = np.arange(9)
b = np.array([[1, 2, 3], [2, 6, 5]])
print(a)
print('reduce:', np.add.reduce(a))
print('accumulate:', np.add.accumulate(a))
print('cumsum：', np.cumsum(a))
print('cumsum:', np.cumsum(b, 1))  # 各行中的数值累加值
print("Reduceat", np.add.reduceat(a, [0, 5, 2, 7]))
print("Reduceat step I", np.add.reduce(a[0:5]))
print("Reduceat step II", a[5])
print("Reduceat step III", np.add.reduce(a[2:7]))
print("Reduceat step IV", np.add.reduce(a[7:]))
print('outer:', np.add.outer(np.arange(3), a))

# 5.9 算术运算： +、-、*
import numpy as np

a = np.array([2, 6, 5])
b = np.array([1, 2, 3])
print(np.add(a, b))       # 加法
print(np.subtract(a, b))  # 减法
print(np.multiply(a, b))  # 乘法

# 5.10 动手实践：数组的除法运算
import numpy as np

a = np.array([2, 6, 5])
b = np.array([1, 2, 3])
c = 3.14 * b
print('divide:', np.divide(a, b), np.divide(b, a))
print('/:', a/b, b/a)
print('true_divide:', np.true_divide(a, b), np.true_divide(b, a))
print('/:', a/b, b/a)
print('floor_divide:', np.floor_divide(a, b), np.floor_divide(b, a), np.floor_divide(c, b))
print('//:', a//b, b//a, c//b)

# 5.12 动手实践：模运算
import numpy as np
a = np.arange(-4, 4)
b = np.arange(-8, 0)
print(a)
print(b)
print('%:', a % b)
print('remainder:', np.remainder(a, b))
print('mod:', np.mod(a, b))
print('fmod:', np.fmod(a, b))
print(np.fmod(1, 3))
print(np.mod(1, 3))

# 5.14 动手实践：计算斐波那契数列,后一个数是前两个数的和。
import numpy as np
F = np.matrix([[1, 1], [1, 0]])
E = np.array([[1, 1], [1, 0]])
print('F:', F)
print("8th Fibonacci:", (F ** 7)[0, 0])
print("8th Fibonacci:", (E ** 7)[0, 0])
# 利用黄金分割公式或通常所说的比奈公式（Binet’ s Formula），加上取整函数，就可以直接计算斐波那契数。
n = np.arange(1, 9)
sqrt5 = np.sqrt(5)
phi = (1 + sqrt5) / 2
fibonacci = np.rint((phi ** n - (-1/phi) ** n) / sqrt5)
print("Fibonacci:", fibonacci)

# 5.15 利萨茹曲线，x = A*sin(a*t + n/2) ，y = B*sin(b*t)。
# 以参数A=B=1、a=9和b=8绘制了利萨茹曲线。
import numpy as np
import matplotlib.pyplot as plt
# import sys

# a = float(sys.argv[1])
# b = float(sys.argv[2])
a = float(input('请输入a:'))
b = float(input('请输入b:'))
t = np.linspace(-np.pi, np.pi, 201)  # pi:3.14...
x = np.sin(a * t + np.pi/2)
y = np.sin(b * t)
plt.plot(x, y)
plt.show()

# 5.17 方波,方波可以近似表示为多个正弦波的叠加。
import numpy as np
import matplotlib.pyplot as plt
# import sys

t = np.linspace(-np.pi, np.pi, 201)
# k = np.arange(1, float(sys.argv[1]))
k = np.arange(1, float(input('请输入k：')))
k = 2 * k - 1
f = np.zeros_like(t)

for i in range(len(t)):
    f[i] = np.sum(np.sin(k * t[i])/k)

f = (4 / np.pi) * f
plt.plot(t, f)
plt.show()

# 5.19 锯齿波和三角波，锯齿波和三角波也是常见的波形。我们也可以将它们表示成无穷傅里叶级数。
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-np.pi, np.pi, 201)
k = np.arange(1, float(input('请输入k：')))
f = np.zeros_like(t)

for i in range(len(t)):
    f[i] = np.sum(np.sin(2 * np.pi * k * t[i]) / k)
f = (-2 / np.pi) * f
plt.plot(t, f, lw=1.0, color=(1, 0, 0))
plt.plot(t, np.abs(f), lw=2.0, color=(0, 1, 0))
plt.show()

# 5.22 动手实践：玩转二进制位
import numpy as np

x = np.arange(-9, 9)
y = -x
print('Sign different?（不同标志）？', (x ^ y) < 0)
print('Sign different?（不同标志）？', np.less(np.bitwise_xor(x, y), 0))
print("Power of 2?\n", x, "\n", (x & (x - 1)) == 0)
print("Power of 2?\n", x, "\n", np.equal(np.bitwise_and(x, (x-1)), 0))
print("Modulus 4\n", x, "\n", x & ((1 << 2) - 1))
print("Modulus 4\n", x, "\n", np.bitwise_and(x, np.left_shift(1, 2) - 1))
