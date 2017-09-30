import scipy.misc as sm
import numpy as np

# Función open_img
def open_img(name, mode):
    data = sm.imread(name, mode)
    return np.matrix(data)

# Función save_img
def save_img(name, data):
    sm.imsave(name, data, "bmp")

# Función normalize_img
def normalize_img(data):
    new_matrix = data
    dim_r, dim_c = data.shape
    for r in range(dim_r):
        for c in range(dim_c):
            new_matrix[r, c] = data[r, c]/255;
    return new_matrix

# Función apply_filter
def apply_filter(r_k, c_k, r, c, data, kernel):
    result = 0
    px_border = r_k//2
    r_d = r - px_border
    for i in range(r_k):
        c_d = c - px_border
        for j in range(c_k):
            result += data[r_d, c_d] * kernel[i, j]
            c_d += 1
        r_d += 1
    return result

# Función filter_img
def filter_img(data, kernel):
    r_k, c_k = kernel.shape
    r_d, c_d = data.shape
    new_matrix = np.zeros((r_d, c_d))
    px_border = r_k//2
    for r in range(r_d):
        for c in range(c_d):
            if c >= px_border and r >= px_border and c_d - c > px_border and r_d - r > px_border:
                new_matrix[r, c] = apply_filter(r_k, c_k, r, c, data, kernel)
    return new_matrix


img = open_img("leena512.bmp", 'L')
norm = normalize_img(img)

kernel_1 = np.matrix([[1, 2, 0, -2, -1],
                      [1, 2, 0, -2, -1],
                      [1, 2, 0, -2, -1],
                      [1, 2, 0, -2, -1],
                      [1, 2, 0, -2, -1]])

kernel_2 = 1/256 * np.matrix([[1, 4, 6, 4, 1],
                              [4, 16, 24, 16, 4],
                              [6, 24, 36, 24, 6],
                              [4, 16, 24, 16, 4],
                              [1, 4, 6, 4, 1]])

test_img = filter_img(norm, kernel_1)
test_img2 = filter_img(norm, kernel_2)

save_img("prueba1", test_img)
save_img("prueba2", test_img2)
