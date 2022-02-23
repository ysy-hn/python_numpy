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