import numpy as np
from scipy.optimize import minimize
import pandas as pd

def new_eta(dxk, dgk, eta):
    temp = np.dot(eta, dgk)
    t = np.dot(dxk - temp, np.transpose(temp)) / np.dot(np.transpose(dgk), temp)
    return eta + t

def pirson(f, grad, x0):
    x_curr = x0
    x_next = x0
    eta = np.eye(2)
    for i in range(3):
        x_curr = x_next
        df = grad(x_curr)
        to_minimize = lambda l: f(x_curr - l * np.dot(eta, df))
        lk = minimize(to_minimize, x0=0, tol=0.03).x[0]
        Sk = -np.dot(eta, df)
        print("k = ", i)
        print("xk: ", x_curr)
        print("f: ", f(x_curr))
        print("df:", df)
        print("eta:", eta)
        print("Sk:", Sk)
        print("")
        x_next = x_curr + lk * Sk
        eta = new_eta(x_next - x_curr, grad(x_next) - grad(x_curr), eta)