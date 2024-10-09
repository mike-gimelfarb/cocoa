import numpy as np
from cocoaopt import Brent


# function to optimize
def fx(x):
    return np.sin(x) + np.sin(10 * x / 3)


alg = Brent(mfev=20000, atol=1e-6)
sol = alg.optimize(fx, guess=3., lower=2.7, upper=7.5)
print(sol)