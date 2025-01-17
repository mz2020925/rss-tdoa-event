# Import packages.
import cvxpy as cp
import numpy as np

# Generate a random feasible SOCP.
m = 3
n = 10
p = 5
n_i = 5
np.random.seed(2)
f = np.random.randn(n)
A = []
b = []
c = []
d = []
x0 = np.random.randn(n)  # 给了待求解变量的初值
for i in range(m):
    A.append(np.random.randn(n_i, n))
    b.append(np.random.randn(n_i))
    c.append(np.random.randn(n))
    d.append(np.linalg.norm(A[i] @ x0 + b, 2) - c[i].T @ x0)
F = np.random.randn(p, n)
g = F @ x0

# Define and solve the CVXPY problem.
x = cp.Variable(n)

# We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
print(A[0] @ x + b[0])
print(c[1].T @ x + d[1])
soc_constraints = [
      cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
]
prob = cp.Problem(cp.Minimize(f.T@x),
                  soc_constraints + [F @ x == g])
prob.solve()

# Print result.
# print("The optimal value is", prob.value)
# print("A solution x is")
# print(x.value)
# for i in range(m):
#     print("SOC constraint %i dual variable solution" % i)
#     print(soc_constraints[i].dual_value)