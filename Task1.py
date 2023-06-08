import pandas as pd
import numpy as np
from IPython.display import display

K1 = 0.3819660112501051
K2 = 0.6180339887498949

def FormBine(n) -> int:
    return (int((0.5 * (1 + np.sqrt(5))) ** n / np.sqrt(5)) if n >= 40 else
    int(((0.5 * (1 + np.sqrt(5))) ** n - (0.5 * (1 - np.sqrt(5))) ** n) / np.sqrt(5)))


def FindFib(value):
    counter = 0
    a = 1
    b = 1
    c = a + b
    while (c < value):
        v = c
        a = b
        b = v
        c = b + a
        counter += 1
    return counter


def DichotomyMethod(f, eps = 1e-3, a0 = -2.0, b0 = 20.0):
    fCount = 0
    a, b, x1, x2 = [], [], [], []
    a.append(a0)
    b.append(b0)
    x1.append((a0 + b0 - eps / 2) / 2)
    x2.append((a0 + b0 + eps / 2) / 2)
    n = 1
    while abs(b[-1] - a[-1]) > eps:
        if f(x1[-1]) <= f(x2[-1]):
            a.append(a[-1])
            b.append(x2[-1])
        else:
            a.append(x1[-1])
            b.append(b[-1])
        x1.append((a[-1] + b[-1] - eps / 2) / 2)
        x2.append((a[-1] + b[-1] + eps / 2) / 2)
        fCount += 2
        n += 1
    df = pd.DataFrame({'ai' : a,
                       'x1' : x1,
                       'x2' : x2,
                       'bi' : b},
                        index = range(1, n + 1))
    return df
    #display(df.style.format('{:.8e}').highlight_min(subset='li', color = '#ACDDDE'))
    
    
    
    
def GoldenRatioMethod(f, eps = 1e-3, a0 = -2.0, b0 = 20.0):
    x1, x2, fx1, fx2, a, b = [], [], [], [], [], []
    fCount = 0
    n = 0
    a.append(a0)
    b.append(b0)
    x1.append(a0 + K1 * (b0 - a0))
    x2.append(a0 + K2 * (b0 - a0))
    fx1.append(f(x1[0]))
    fx2.append(f(x2[0]))
    fCount += 2
    while True:
        n += 1
        if fx1[-1] <= fx2[-1]:
            a.append(a[-1])
            b.append(x2[-1])
            x2.append(x1[-1])
            x1.append(a[-1] + K1 * (b[-1] - a[-1]))
            fx2.append(fx1[-1])
            fx1.append(f(x1[-1]))
        else:
            a.append(x1[-1])
            b.append(b[-1])
            x1.append(x2[-1])
            x2.append(a[-1] + K2 * (b[-1] - a[-1]))
            fx1.append(fx2[-1])
            fx2.append(f(x2[-1]))
        fCount += 1
        if abs(b[-1] - a[-1]) <= eps:
            break
    arr = []
    arr.append(np.NaN)
    for i in range(1, n + 1):
        arr.append((b[i - 1] - a[i - 1]) / (b[i] - a[i]))
    df = pd.DataFrame({'ai' : a,
                       'x1' : x1,
                       'x2' : x2,
                       'bi' : b},
                        index = range(1, n + 2))
    return df                    
    #display(df.style.format('{:.8e}').highlight_min(subset='li', color = '#ACDDDE'))
    

    
    
def FibonachiMethod(f, eps = 1e-3, a0 = -2.0, b0 = 20.0):
    a, b, x1, x2, fx1, fx2 = [], [], [], [], [], []
    fCount = 0
    delt = b0 - a0
    n2 = FindFib(delt / eps)
    Fn2 = FormBine(n2)
    n = 0
    a.append(a0)
    b.append(b0)
    x1.append(a[0] + (b[0] - a[0]) * FormBine(n2 - 2) / Fn2)
    x2.append(a[0] + b[0] - x1[0])
    fx1.append(f(x1[0]))
    fx2.append(f(x2[0]))
    fCount += 2
    while True:
        n += 1
        if fx1[-1] <= fx2[-1]:
            a.append(a[-1])
            b.append(x2[-1])
            x2.append(x1[-1])
            x1.append(a[-1] + delt * FormBine(n2 - n - 1) / Fn2)
            fx2.append(fx1[-1])
            fx1.append(f(x1[-1]))
        else:
            a.append(x1[-1])
            b.append(b[-1])
            x1.append(x2[-1])
            x2.append(a[-1] + delt * FormBine(n2 - n) / Fn2)
            fx1.append(fx2[-1])
            fx2.append(f(x2[-1]))
        fCount += 1
        if abs(b[-1] - a[-1]) <= eps:
            break
    arr = []
    arr.append(np.NaN)
    for i in range(1, n):
        arr.append((b[i - 1] - a[i - 1]) / (b[i] - a[i]))
    arr.append(np.NaN)
    df = pd.DataFrame({'ai' : a,
                       'x1' : x1,
                       'x2' : x2,
                       'bi' : b},
                        index = range(1, n + 2))
    return df