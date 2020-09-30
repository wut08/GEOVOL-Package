import numpy as np

def get_x(e_square):
    x_est = np.sum(e_square ** 2, axis=1) / e_square.shape[1]
    st_x = x_est / np.mean(x_est)

    return st_x

