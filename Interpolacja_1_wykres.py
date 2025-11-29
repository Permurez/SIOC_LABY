import numpy as np
import matplotlib.pyplot as plt

# Parametry
a = 3  # wybór jądra
c = [2, 4, 10]  # współczynniki mnożące liczbę punktów
pi = np.pi
N_org = 100
x = np.linspace(-pi, pi, N_org)
wybor_funkcji = 1  # wybór funkcji

# Definicje jąder
def h1(t):
    return np.where((t >=0) & (t<=1), 1, 0)

def h2(t):
    return np.where((t >=-0.5) & (t <=0.5), 1, 0)

def h3(t):
    return np.where((t >= -1) & (t <= 1), 1 - np.abs(t), 0)

def h4(t):
    return np.where(t !=0, np.sin(t)/t, 1)

jadra = {1: h1, 2: h2, 3: h3, 4: h4}

# Definicje funkcji do interpolacji
def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(1/x)

def f3(x):
    return np.sign(np.sin(8 * x))

funkcje = {1: f1, 2: f2, 3: f3}
y = funkcje[wybor_funkcji](x)

# Funkcja obliczająca błąd średniokwadratowy
def MSE(y_org, y_new):
    N = len(y_org)
    if N == 0:
        return 0
    else:
        return np.sum((y_org - y_new)**2)/N

# Funkcja interpolacyjna
def interpolacja(x_org, y_org, x_new, j):
    dx = x_org[1] - x_org[0]
    t = (x_new[:, None] - x_org[None, :]) / dx
    waga = j(t)
    y_new = np.sum(y_org * waga, axis=1) / np.sum(waga, axis=1)
    return y_new


fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, mult in enumerate(c):
    N_new = N_org * mult
    x_new = np.linspace(-pi, pi, N_new)
    y_new = interpolacja(x, y, x_new, jadra[a])
    mse = MSE(funkcje[wybor_funkcji](x_new), y_new)

    axs[i].plot(x, y, 'k.', label='oryginał')
    axs[i].scatter(x_new, y_new, color='r', label=f'{mult} razy więcej punktów')
    axs[i].set_title(f'MSE={mse:.20f}')#4 do wykresow, do tabeli wiecej
    axs[i].legend()
    axs[i].grid()

fig.suptitle(f'Interpolacja funkcji f{wybor_funkcji} jądrem h{a}', fontsize=16)
plt.tight_layout()
plt.show()