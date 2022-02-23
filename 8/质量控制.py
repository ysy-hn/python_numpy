# 知识
# TDD主要专注于自动单元测试，它的目标是尽最大限度自动化测试代码。如果代码被改动，
# 我们仍可以运行测试并捕捉可能存在的问题。换言之，测试对于已经存在的功能模块依然有效。
# 8.1 断言函数，单元测试通常使用断言函数作为测试的组成部分。
# assert_almost_equal：如果两个数字的近似程度没有达到指定精度，就抛出异常
# assert_approx_equal：如果两个数字的近似程度没有达到指定有效数字，就抛出异常
# assert_array_almost_equal：如果两个数组中元素的近似程度没有达到指定精度，就抛出异常
# assert_array_equal：如果两个数组对象不相同，就抛出异常
# assert_array_less：两个数组必须形状一致，并且第一个数组的元素严格小于第二个数组的元素，否则就抛出异常
# assert_equal：如果两个对象不相同，就抛出异常
# assert_raises：若用填写的参数调用函数没有抛出指定的异常，则测试不通过
# assert_warns：若没有抛出指定的警告，则测试不通过
# assert_string_equal：断言两个字符串变量完全相同
# assert_allclose：如果两个对象的近似程度超出了指定的容差限，就抛出异常

# 8.2 动手实践：
# assert_almost_equal函数：断言近似相等。

# 8.4 动手实践：使用 assert_approx_equal 断言近似相等
# assert_approx_equal函数：如果两个数字的近似程度没有达到指定的有效数字要求，将抛出异常。

# 8.5 数组近似相等
# assert_array_almost_equal函数：两个数组中元素的近似程度没有达到指定的精度要求，将抛出异常；
# 形状是否一致，然后逐一比较两个数组中的元素。

# 8.7 数组相等
# assert_array_equal函数：如果两个数组对象不相同，将抛出异常；
# 两个数组相等必须形状一致且元素也严格相等，允许数组中存在NaN元素。
# assert_allclose函数：该函数有参数atol（绝对容差限）和rtol（相对容差限）。
# 对于两个数组a和b，将测试是否满足以下等式：|a - b| <= (atol + rtol * |b|)

# 8.9 数组排序
# assert_array_less函数：两个数组必须形状一致并且第一个数组的元素严格小于第二个数组的元素，否则将抛出异常。

# 8.11 对象比较
# assert_equal函数：两个对象不相同，将抛出异常。对象为NumPy数组、列表、元组或字典。

# 8.13 字符串比较
# assert_string_equal函数：断言两个字符串变量完全相同。该函数区分字符大小写。

# 8.15 浮点数比较，ULP是浮点数的最小精度单位。
# 机器精度（machine epsilon）是指浮点运算中的相对舍入误差上界。
# 机器精度等于ULP相对于1的值。NumPy中的finfo函数可以获取机器精度。

# 8.16 动手实践
# assert_array_almost_equal_nulp函数：比较浮点数。

# 8.17 多ULP的浮点数比较
# assert_array_max_ulp函数：可以指定ULP的数量作为允许的误差上界。
# 参数maxulp接受整数作为ULP数量的上限，默认值为1。

# 8.19 单元测试
# 单元测试是对代码的一小部分进行自动化测试的单元，通常是一个函数或方法。
# Python中有用于单元测试的PyUnit API（应用程序编程接口）。
# 编写一个包含单元测试的类，继承Python标准库unittest模块中的TestCase类。

# 8.21 nose 和测试装饰器
# numpy.testing.decorators.deprecated：在运行测试时过滤掉过期警告
# numpy.testing.decorators.knownfailureif：根据条件抛出KnownFailureTest异常
# numpy.testing.decorators.setastest：将函数标记为测试函数或非测试函数
# numpy.testing.decorators. skipif：根据条件抛出SkipTest异常
# numpy.testing.decorators.slow：将测试函数标记为“运行缓慢”
# decorate_methods函数，将装饰器应用到能够匹配正则表达式或字符串的类方法上。
# 安装nose：pip install nose
# skipif装饰器跳过测试；
# knownfailureif装饰器使得该测试总是不通过；

# 8.23 文档字符串
# 文档字符串（docstring）是内嵌在Python代码中的类似交互式会话的字符串。
# 这些字符串可以用于某些测试，也可以仅用于提供使用示例。
# numpy.testing模块的rundocs函数，从而执行文档字符串测试。



# 例题
# 8.2 assert_almost_equal 断言近似相等
import numpy as np
print('decimal(十进制）第8位：', np.testing.assert_almost_equal(0.123456789, 0.123456780,
                                                      decimal=8))
print('decimal(十进制）第9位：', np.testing.assert_almost_equal(0.123456789, 0.123456780,
                                                      decimal=9))

# 8.4 assert_approx_equal 断言近似相等
import numpy as np
print('significant(重要）第8位:', np.testing.assert_approx_equal(0.123456789, 0.123456780,
                                                            significant=8))
print('significant(重要）第9位:', np.testing.assert_approx_equal(0.123456789, 0.123456780,
                                                            significant=9))

# 8.6 assert_array_almost_equal 数组近似相等
import numpy as np
print('8:', np.testing.assert_array_almost_equal([0, 0.123456789], [0, 0.123456780],
                                           decimal=8))
print('9', np.testing.assert_array_almost_equal([0, 0.123456789], [0, 0.123456780],
                                           decimal=9))

# 8.8 比较数组,assert_array_equal/assert_allclose
import numpy as np
print("pass:", np.testing.assert_array_equal([0, 0.123456789, np.nan],
                                             [0, 0.123456789, np.nan]))
print("Fail:", np.testing.assert_array_equal([0, 0.123456789, np.nan],
                                             [0, 0.123456780, np.nan]))
print("Pass:", np.testing.assert_allclose([0, 0.123456789, np.nan],
                                          [0, 0.123456780, np.nan], rtol=1e-7, atol=0))
print("fail:", np.testing.assert_allclose([0, 0.123456789, np.nan],
                                          [0, 0.123456780, np.nan], rtol=1e-8, atol=0))

# 8.9 核对数组排序 assert_array_less
import numpy as np
print('pass:', np.testing.assert_array_less([0, 0.123456789, np.nan],
                                            [1, 0.23456780, np.nan]))
print('fail:', np.testing.assert_array_less([0, 0.123456789, np.nan],
                                            [0, 0.23456780, np.nan]))

# 8.12 比较对象, assert_equal
import numpy as np
print("Equal(相等）?", np.testing.assert_equal((1, 2), (1, 3)))

# 8.13 字符串比较, assert_string_equal
import numpy as np
print(np.testing.assert_string_equal("NumPy", "NumPy"))
print(np.testing.assert_string_equal("NumPy", "Numpy"))

# 8.15 浮点数比较
# finfo函数:确定机器精度;
# assert_array_almost_equal_nulp
import numpy as np
eps = np.finfo(float)
print(eps)
eps = np.finfo(float).eps
print(eps)
print('pass:', np.testing.assert_array_almost_equal_nulp(1, 1+eps))
print('fail:', np.testing.assert_array_almost_equal_nulp(1, 1+2*eps))

# 8.17 多 ULP 的浮点数比较,
# assert_array_max_ulp函数:指定ULP的数量作为允许的误差上界。
# 参数maxulp接受整数作为ULP数量的上限，默认值为1。
import numpy as np
eps = np.finfo(float).eps
print('pass:', np.testing.assert_array_max_ulp(1, 1+eps))
print('pass:', np.testing.assert_array_max_ulp(1, 1+2*eps, maxulp=2))

# 8.20 动手实践：编写单元测试
import numpy as np
import unittest

def factorial(n):
    if n == 0:
        return 1
    if n < 0:
        raise ValueError
    return np.arange(1, n+1).cumprod()

class FactorialTest(unittest.TestCase):
    def test_factorial(self):
        """计算3的阶乘，测试通过"""
        self.assertEqual(6, factorial(3)[-1])
        np.testing.assert_equal(np.array([1, 2, 6]), factorial(3))

    def test_zero(self):
        """计算0的阶乘，测试通过"""
        self.assertEqual(1, factorial(0))

    def test_negative(self):
        """计算负数的阶乘，测试不通过,这里应抛出ValueError异常，但我们断言其抛出IndexError异常"""
        self.assertRaises(IndexError, factorial(-10))

if __name__ == '__main__':
    unittest.main()

# 8.24 动手实践：执行文档字符串测试
import numpy as np

def factorial(n):
    def factorial(n):
        """
        Test for the factorial of 3 that should pass.
        >>> factorial(3)
        6
        Test for the factorial of 0 that should fail.
        >>> factorial(0)
        1
        """
    return np.arange(1, n + 1).cumprod()[-1]
















