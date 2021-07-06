import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import derivative
from scipy.integrate import solve_ivp


def peaks(x, y):
    q = -1
    r0 = [0.0, 0.0]
    return q / np.hypot((x - r0[0]), (y - r0[1]))


X, Y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
Z = peaks(X, Y)


def peaks_x(x, y):
    return derivative(lambda x0: peaks(x0, y), x, 0.1)


def peaks_y(x, y):
    return derivative(lambda y0: peaks(x, y0), y, 0.1)


def grad(t, xy):
    x = xy[0]
    y = xy[1]
    return peaks_x(x, y), peaks_y(x, y)


t = [-10, 10]
y0 = [10.0, 0.0]
sol = solve_ivp(grad, t, y0, t_eval=[0, 5])
xs = sol.t
ys = sol.y[1]

print(sol)

dx, dy = np.gradient(Z)
dx = 1.0 / dx
dy = 1.0 / dy

for i in range(len(X)):
    for j in range(len(Y)):
        if X[i][j] < 0 and Y[i][j] < 0:
            dx[i][j] = -dx[i][j]
            dy[i][j] = -dy[i][j]
        if X[i][j] > 0 and Y[i][j] > 0:
            dx[i][j] = -dx[i][j]
            dy[i][j] = -dy[i][j]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
ax.set(xlim=(-5, 5), ylim=(-5, 5))
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_aspect('equal')

levels = np.linspace(np.min(Z), np.max(Z), 10)
# ax.contour(X, Y, Z, linewidths=1, linestyles='solid',
#             colors='k', levels=levels)
# ax.contourf(X, Y, Z, levels=levels, cmap=plt.get_cmap('coolwarm'))
ax.contourf(X, Y, Z, levels=levels, cmap=plt.get_cmap('coolwarm'))
ax.arrow(xs[0], ys[0], xs[1], ys[1], head_width=0.2, head_length=0.2, length_includes_head=True)

dz = np.sqrt(dx**2 + dy**2)
ax.quiver(X, Y, dx / dz, dy / dz)
