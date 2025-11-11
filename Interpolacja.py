import numpy as np
import matplotlib.pyplot as plt
#A to wybór jądra, C to wspłoczynnik ilości punktów
a=4
c= [10]
pi=np.pi
N_org= 10
x= np.linspace(-pi, pi, N_org)
#y= np.sin(x**(-1))
y=np.tan(np.sin(x))
def j1(t):
    return np.where((t >=0) & (t<=1), 1, 0)
def j2(t):
    return np.where((t >=-0.5) & (t <=0.5), 1,0)
def j3(t):
    return np.where((t >= -1) & (t <= 1), 1 - np.abs(t), 0)
def j4(t):
    return np.where (t !=0, np.sin(t)/t, 1)

jadra = {1: j1, 2: j2, 3: j3, 4: j4}
plt.figure(figsize=(12, 8))

def MSE(y_org, y_new):
    N=len(y_org)
    if N==0:
        return 0
    else:
        mse = np.sum((y_org - y_new)**2)/N
    return mse

def interpolacja(x_org, y_org, x_new, j):
    dx=x_org[1]-x_new[0]
    t=(x_new[:, None] - x_org[None, :]) / dx
    waga=j(t)
    y_new = np.sum(y_org * waga, axis=1) / np.sum(waga, axis=1)
    return y_new

for kernel_num in [a]:
    plt.scatter(2, 2, kernel_num)
    plt.plot(x, y, 'k.', label='oryginał')

    for mult in c:
        N_new = N_org * mult
        x_new = np.linspace(-pi, pi, N_new)
        y_new = interpolacja(x, y, x_new, jadra[kernel_num])
        mse = MSE(np.sin(x_new), y_new)
        plt.scatter(x_new, y_new, label=f'{mult} punktów, MSE={mse:.4f}')

    plt.title(f'Interpolacja jądrem j{kernel_num}')
    plt.legend()
    plt.grid()
   
plt.tight_layout()
plt.show()