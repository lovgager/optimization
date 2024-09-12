import numpy as np
import matplotlib.pyplot as plt


def bisect(a, b, eps, f):
# returns argmin(f), min(f) and number of iterations 
    c = (a + b)/2
    fc = f(c) 
    for k in range(1000): # [a, y, c, z, b]
        y = (a + c)/2
        z = (c + b)/2
        fy = f(y)
        fz = f(z)
        if fy < fc: # remove (c, b]
            b = c
            c = y
            fc = fy
        elif fz < fc: # remove [a, c)
            a = c
            c = z
            fc = fz
        else: # remove [a, y), (z, b]
            a = y
            b = z
        if b - a <= eps:
            x = (a + b)/2
            return x, f(x), 2*(k+1)
    return 1/0 # error


def bisect2d(ax, bx, ay, by, eps, f, x0, y0):
# returns argmin(f), min(f) and number of iterations 
    n = 0
    for _ in range(1000):
        f_y = lambda y: f(x0, y)
        y, _, ny = bisect(ay, by, eps, f_y)
        f_x = lambda x: f(x, y)
        x, _, nx = bisect(ax, bx, eps, f_x)
        n += ny + nx
        if (x - x0)**2 + (y - y0)**2 < eps**2:
            return x, y, f(x, y), n
        x0, y0 = x, y
    return 1/0 # error
    


def bruteforce2d(ax, bx, ay, by, eps, f):
# returns argmin(f), min(f) and number of iterations 
    nx = int((bx - ax)//eps)
    ny = int((by - ay)//eps)
    x = np.linspace(ax, bx, nx + 1)
    y = np.linspace(ay, by, ny + 1)
    xmin, ymin = ax, ay
    fmin = f(ax, ay)
    fs = np.zeros((nx + 1, ny + 1))
    for i in range(nx + 1):
        for j in range(ny + 1):
            fs[i, j] = f(x[i], y[j])
    i, j = np.unravel_index(np.argmin(fs), fs.shape)
    xmin, ymin = x[i], y[j]
    fmin = f(xmin, ymin)
    return xmin, ymin, fmin, (nx + 1)*(ny + 1)


def gradient_descent(ax, bx, ay, by, eps, f, g, x0, y0):
# returns argmin(f), min(f) and number of iterations
    x, y = x0, y0
    gx, gy = g(x, y)
    alpha = 0.1
    for k in range(10000):
        x_next = x - alpha*gx
        y_next = y - alpha*gy
        x_next = max([x_next, ax])
        x_next = min([x_next, bx])
        y_next = max([y_next, ay])
        y_next = min([y_next, by])
        if f(x_next, y_next) > f(x, y): 
            alpha /= 2
        else: 
            if (x_next, y_next) == (x, y): break
            x, y = x_next, y_next
            gx, gy = g(x, y)
        if gx**2 + gy**2 < eps**2: break
    return x, y, f(x, y), k + 1


def plot_contour(f):
    N = 101
    x = np.linspace(0, 10, N)
    y = np.linspace(0, 10, N)
    X, Y = np.meshgrid(x, y)
    plt.contour(X, Y, f(X, Y), levels=[10, 25, 100, 250, 500, 1000, 2500, 5000, 10000])
    plt.scatter([9.5], [9.5], c='red', marker='x', s=100)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('fe(x)')
    

ax = ay = 0
bx = by = 10
eps = 0.001
x0, y0 = 1, 2

f_a = lambda x, y: (x - 3)**2 + y**2 + 1
f_b = lambda x, y: (x + 1)**2 + (2*y - 8)**2 + 1
f_c = lambda x, y: x**2 + x*y + y**2 - 16*x - 17*y + 94
f_d = lambda x, y: ((x - 7)**2 + (y - 8)**2)*((x - 8)**2 + (y - 7)**2) + 4
f_e = lambda x, y: (4*x*y-19)**2*(np.cos(np.pi*x)**2+np.cos(np.pi*y)**2) - x - y + 24
functions = [f_a, f_b, f_c, f_d, f_e]

fg_a = lambda x, y: [2*(x-3), 2*y]
fg_b = lambda x, y: [2*(x+1), 4*(2*y-8)]
fg_c = lambda x, y: [2*x +y-16, x+2*y-17]
fg_d = lambda x, y: [2*(x-7)*((x-8)**2+(y-7)**2)+((x-7)**2+(y-8)**2)*2*(x-8),
                              2*(y-8)*((x-8)**2+(y-7)**2)+((x-7)**2+(y-8)**2)*2*(y-7)]
fg_e = lambda x, y: [8*y*(4*x*y-19)*(np.cos(np.pi*x)**2+np.cos(np.pi*y)**2) + 
                              (4*x*y-19)**2*(-np.pi)*np.sin(2*np.pi*x)-1,
                              8*x*(4*x*y-19)*(np.cos(np.pi*x)**2+np.cos(np.pi*y)**2) + 
                              (4*x*y-19)**2*(-np.pi)*np.sin(2*np.pi*y)-1,]
grads = [fg_a, fg_b, fg_c, fg_d, fg_e]


print('\nBrute force 2D:\n')
for f in functions:
    res = bruteforce2d(ax, bx, ay, by, eps, f)
    print('x* = {0:0.4f}, y* = {1:0.4f}, f(x*,y*) = {2:0.4f}, n = {3}'.format(*res))
    
print('\nAlternating directions with bisection 1D:\n')
for f in functions:
    res = bisect2d(ax, bx, ay, by, eps, f, x0, y0)
    print('x* = {0:0.4f}, y* = {1:0.4f}, f(x*,y*) = {2:0.4f}, n = {3}'.format(*res))
    
print('\nGradient descent 2D:\n')
for f, g in zip(functions, grads):
    res = gradient_descent(ax, bx, ay, by, eps, f, g, x0, y0)
    print('x* = {0:0.4f}, y* = {1:0.4f}, f(x*,y*) = {2:0.4f}, n = {3}'.format(*res))
