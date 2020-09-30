import numpy as np
from factor_analyzer import FactorAnalyzer

def get_s(e_square):
    
    fa = FactorAnalyzer(n_factors=1, rotation = "varimax")
    fa.fit(e_square)
    loadings = fa.loadings_

    tmp = (loadings - min(loadings)) / (max(loadings) - min(loadings))
    s = tmp / sum(tmp ** 2)
    s_prime = s / np.linalg.norm(s)

    return s_prime

