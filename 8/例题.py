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
