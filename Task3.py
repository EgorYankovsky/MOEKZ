import numpy as np
from scipy.optimize import minimize
import numdifftools as nd

def pirson(f, eta_alg, x0):
    x_curr = x0
    x_next = x0
    eta = np.eye(2)
    for i in range(3):
        x_curr = x_next
        df = nd.Gradient(f)(x_curr)
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
        eta = eta_alg(x_next - x_curr, nd.Gradient(f)(x_next) - nd.Gradient(f)(x_curr), eta)