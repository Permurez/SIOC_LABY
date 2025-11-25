import numpy as np
import matplotlib.pyplot as plt
#A to wybór jądra, C to wspłoczynnik ilości punktów
a=1
c= [2,4,10]
pi=np.pi
N_org= 100
x= np.linspace(-pi, pi, N_org)
wybor_funkcji = 2
#y= np.sin(x**(-1))
y=np.sin(x)
def h1(t):
    return np.where((t >=0) & (t<=1), 1, 0)
def h2(t):
    return np.where((t >=-0.5) & (t <=0.5), 1,0)
def h3(t):
    return np.where((t >= -1) & (t <= 1), 1 - np.abs(t), 0)
def h4(t):
    return np.where (t !=0, np.sin(t)/t, 1)

jadra = {1: h1, 2: h2, 3: h3, 4: h4}
plt.figure(figsize=(12, 8))
def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(1/x)
def f3(x):
    return np.exp(-x**2)

funkcje = {1: f1, 2: f2, 3: f3}
 
y = funkcje[wybor_funkcji](x)

def MSE(y_org, y_new):
    N=len(y_org)
    if N==0:
        return 0
    else:   
        mse = np.sum((y_org - y_new)**2)/N
    return mse

def interpolacja(x_org, y_org, x_new, j):
    dx=x_org[1]-x_org[0]
    t=(x_new[:, None] - x_org[None, :]) / dx
    waga=j(t)
    y_new = np.sum(y_org * waga, axis=1) / np.sum(waga, axis=1)
    return y_new

for kernel_num in [a]:
 
    plt.plot(x, y, 'k.', label='oryginał')  
    
    for mult in c:
        N_new = N_org * mult
        x_new = np.linspace(-pi, pi, N_new)
        y_new = interpolacja(x, y, x_new, jadra[kernel_num])
        mse = MSE(funkcje[wybor_funkcji](x_new), y_new)
        plt.scatter(x_new, y_new, label=f'{mult} razy więcej punktów, MSE={mse:.4f}')
        

    plt.title(f'Interpolacja funkcji f{wybor_funkcji} jądrem h{kernel_num}')
    plt.legend()
    plt.grid()
   
plt.tight_layout()

plt.show()