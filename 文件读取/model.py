import numpy as np
import cvxpy as cp

if __name__ == '__main__':
    n = eval(input())
    A = []
    for i in range(n):
        A.append(list(map(int, input().split(','))))
    A = np.array(A)

    num = np.array(list(map(int, input().split(','))))
    print(A)

    x =