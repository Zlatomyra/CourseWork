import numpy as np

def recurrence_relations(C1, C2, n, m):
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    b = np.zeros((n + 1, n + 2 * m + 1))
    z = np.zeros((n, m))

    for k in range(n - 1):
        for i in range(k + 1, n):
            numerator = C1[i, k]
            for j in range(k):
                numerator -= C1[i, j] * u[j, k]

            denominator = C1[k, k]
            for j in range(k):
                denominator -= C1[k, j] * u[j, k]

            b[i + 1, k + 1] = numerator / denominator

        v[k, k] = b[k + 2, k + 1]

        for s in range(k - 1, -1, -1):
            sum_ = 0.0
            for i in range(s + 1, k + 1):
                sum_ += b[i + 1, s + 1] * v[i, k]
            v[s, k] = b[k + 2, s + 1] - sum_

    for k in range(n - 1):
        for i in range(k + 1, n + m):
            numerator = C2[k, i]
            for j in range(k):
                numerator -= C2[j, i] * v[j, k]

            denominator = C2[k, k]
            for j in range(k):
                denominator -= C2[j, k] * v[j, k]

            b[k + 1, i + 1] = numerator / denominator

        u[k, k] = b[k + 2, k + 1]

        for s in range(k - 1, -1, -1):
            sum_ = 0.0
            for i in range(s + 1, k + 1):
                sum_ += b[s + 1, i + 1] * u[i, k]
            u[s, k] = b[s + 1, k + 2] - sum_

    for i in range(n):
        for j in range(m):
            val = b[i + 1, m + j + 1]
            for k in range(n - 1):
                val -= b[i + 1, m + k + 1] * z[k, j]
            z[i, j] = val

    return z

def check_error(check, index, tol=1e-2):
    norm_val = np.linalg.norm(check)
    print(f"\nПеревірка для вектора p^({index}):")
    print(f"  - Норма похибки: {norm_val:.3e}")
    if norm_val < tol:
        print(f"    Перевірка пройдена успішно (норма < {tol:.0e})")
    else:
        print(f"    Є відхилення (норма >= {tol:.0e})")


def results(C11, C12, C21, C22):
    Z1 = recurrence_relations(C11.T, C21.T, n, m)
    Z2 = recurrence_relations(C22.T, C12.T, m, n)

    rhs1 = k1 + Z1 @ k2
    rhs2 = k2 + Z2 @ k1

    p1 = np.linalg.solve(C11.T, rhs1)
    p2 = np.linalg.solve(C22.T, rhs2)

    return p1, p2

n = 2
m = 3

A11 = np.array([[0.2, 0.1],
                 [0.4, 0.3]])

A12 = np.array([[0.1, 0.2, 0.1],
                 [0.2, 0.1, 0.3]])

A21 = np.array([[0.05, 0.1 ],
                 [0.1, 0.05],
                 [0.04, 0.03]])

A22 = np.array([[0.05, 0.02, 0.01],
                 [0.03, 0.04, 0.02],
                 [0.01, 0.03, 0.05]])

k1 = np.array([1.0, 1.0])
k2 = np.array([0.5, 0.5, 0.3])

En = np.eye(2)
Em = np.eye(3)

C11 = A11.T
C12 = A21.T
C21 = A12.T
C22 = A22.T

p1, p2 = results(C11, C12, C21, C22)

print("\nМатриця A11 — матриця затрат продукції i на випуск одиниці продукції j")
print(A11)

print("\nМатриця A12 — матриця затрат продукції і на знищення одиниці забруднювача")
print(A12)

print("\nМатриця A21 — матриця забруднювача l під час випуску одиниці продукції j ")
print(A21)

print("\nМатриця A22 — матриця випуску забруднювача l під час знищення одиниці забруднювача s")
print(A22)

print("\nВектор k^(1) — вектор коефіцієнтів додаткової вартості продукції основного виробництва")
print(k1)

print("\nВектор k^(2) — вектор коефіцієнтів додаткової вартості продукції допоміжного виробництва")
print(k2)

print("\nВектор p^(1) — вектор цін основної продукції ")
print("p^{(1)} =\n", p1)

print("\nВектор p^(2) — вектор вартостей знищення одиниць забруднювачів")
print("p^{(2)} =\n", p2)

check_p1 = C11 @ p1 + C12 @ p2 - k1
check_p2 = C21 @ p1 + C22 @ p2 - k2

check_error(check_p1, 1)
check_error(check_p2, 2)
