import numpy as np

# 鸡兔同笼问题参数
n = 35  # 总头数
m = 94  # 总脚数

# 构造系数矩阵 A 和常数向量 B
# “x“=鸡， y=”兔子“， x+y = n, 2x+4y = m
A = np.array([[1, 1], [2, 4]])
B = np.array([n, m])

# 解线性方程组
solution = np.linalg.solve(A, B)
chickens, rabbits = solution

print(f"鸡的数量: {int(chickens)}")
print(f"兔子的数量: {int(rabbits)}")
