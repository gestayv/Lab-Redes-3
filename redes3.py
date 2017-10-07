import scipy.misc as sm
import scipy.fftpack as sft
import matplotlib.pyplot as plt
import numpy as np
import time

# Función open_img: Se encarga de abrir un archivo de imagen.
# Entrada:  - name: Nombre de la imagen, se incluye su extensión.
# Salida:   Matriz que representa el archivo que se abre, donde cada elemento de ella corresponde a un pixel de la
# imagen.
def open_img(name):
    data = sm.imread(name, flatten=True)
    return data

# Función save_img: Se encarga de guardar un archivo de imagen en el equipo
# Entrada:  - name: Nombre con el que se almacena la nueva imagen.
#           - data: Matriz que representa la imagen a ser guardada, donde cada elemento es un pixel.
# Salida:   La función no genera una salida.
def save_img(name, data):
    sm.imsave(name, data, "bmp")

# Función normalize_img: Se encarga de normalizar los datos de la matriz que representa la imagen que se procesa,
# dejando todos sus valores entre 0 y 1.
# Entrada:  - data: Matriz que representa la imagen que se desea procesar.
# Salida:   Matriz que representa la imagen, donde cada elemento de ella es un pixel y cuyo valor está en el rango
# [0, 1].
def normalize_img(data):
    new_matrix = data
    dim_r, dim_c = (len(data), len(data[0]))
    for r in range(dim_r):
        for c in range(dim_c):
            new_matrix[r][c] = data[r][c]/255
    return new_matrix

# Función apply_filter: Función que se encarga de aplicar el kernel en una región de la imagen original.
# Entrada:  - r_k, c_k: Dimensiones de la matriz kernel.
#           - r, c:     Posición actual en la matriz que representa la imagen sobre la que se aplica el filtro.
#           - data:     Matriz correspondiente a la imagen sobre la que se aplica el filtro.
#           - kernel:   Matriz correspondiente al kernel que se emplea para aplicar el filtro.
# Salida:   Valor correspondiente a la suma ponderada del kernel por el pixel (r, c) y sus pixeles adyacentes.
def apply_filter(r_k, c_k, r, c, data, kernel):
    result = 0
    px_border = r_k//2
    r_d = r - px_border
    rows, columns = (len(data), len(data[0]))
    for i in range(r_k):
        c_d = c - px_border
        for j in range(c_k):
            if 0 <= c_d < columns and 0 <= r_d < rows:
                result += (data[r_d][c_d] * kernel[i][j])
            c_d += 1
        r_d += 1
    return result

# Función filter_img: Se encarga de recorrer la matriz que representa la imagen que se filtra, de manera que se pueda
# aplicar el filtro sobre todos los pixeles de la matriz.
# Entrada:  - data: Matriz que representa la imagen a ser filtrada.
#           - kernel: Matriz kernel, que se emplea para aplicar el filtro sobre cada pixel de la imagen original.
# Salida:   Matriz que representa la imagen con el filtro ya aplicado.
def filter_img(data, kernel):
    r_k, c_k = (len(kernel), len(kernel[0]))
    r_d, c_d = (len(data), len(data[0]))
    new_matrix = np.zeros((r_d, c_d)).tolist()
    for r in range(r_d):
        for c in range(c_d):
            new_matrix[r][c] = apply_filter(r_k, c_k, r, c, data, kernel)
    return new_matrix

# Función filter_gauss: Se encarga de aplicar un filtro gaussiano sobre una imagen indicada.
# Entrada:  - data: Matriz que representa la imagen a ser filtrada, normalizada entre 0 y 1.
# Salida:   Matriz, donde cada elemento que la compone corresponde a un pixel de la imagen luego de que se le ha
#           aplicado el filtro.
def filter_gauss(data):
    kernel = 1/256 * np.matrix([[1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]])
    kernel = kernel.tolist()
    new_img = filter_img(data, kernel)
    return new_img

# Función filter_border: Se encarga de aplicar un filtro detector de bordes sobre una imagen indicada.
# Entrada:  - data: Matriz que representa la imagen a ser filtrada, normalizada entre 0 y 1.
# Salida:   Matriz, donde cada elemento que la compone corresponde a un pixel de la imagen luego de que se le ha
#           aplicado el filtro.
def filter_border(data):
    kernel = np.matrix([[1, 2, 0, -2, -1],
                        [1, 2, 0, -2, -1],
                        [1, 2, 0, -2, -1],
                        [1, 2, 0, -2, -1],
                        [1, 2, 0, -2, -1]])
    kernel = kernel.tolist()
    new_img = filter_img(data, kernel)
    return new_img

# Función fourier_transform: Se encarga de graficar tres transformadas de Fourier 2d (una imagen en su estado original,
# una imagen que pasó por un filtro gaussiano y una imagen que pasó por un filtro detector de bordes).
# Entrada:  - data_1: Matriz de datos, donde cada elemento corresponde a un pixel de la imagen original.
#           - data_2: Matriz de datos, donde cada elemento corresponde a un pixel de la imagen pasada por un filtro
#                     gaussiano.
#           - data_3: Matriz de datos, donde cada elemento corresponde a un pixel de la imagen pasada por un filtro
#                     detector de bordes.
# Salida:   La función genera una imagen con los gráficos de las tres transformadas.
def fourier_transform(data_1, data_2, data_3):
    fft_original = sft.fftshift(sft.fft2(data_1))
    spec_orig = np.log(np.abs(fft_original))

    fft_gauss = sft.fftshift(sft.fft2(data_2))
    spec_gauss = np.log(np.abs(fft_gauss))

    fft_border = sft.fftshift(sft.fft2(data_3))
    spec_border = np.log(np.abs(fft_border))

    plt.figure(figsize=(10.24, 7.20), dpi=100)

    plt.subplot(221)
    plt.title("Transformada de la imagen original")
    freq_x = np.fft.fftfreq(spec_orig.shape[0], d=1 / (2 * spec_orig.max()))
    freq_y = np.fft.fftfreq(spec_orig.shape[1], d=1 / (2 * spec_orig.max()))
    plt.imshow(spec_orig, cmap='hsv', extent=(freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()))
    plt.ylabel("u (frecuencias verticales)")
    plt.xlabel("v (frecuencias horizontales)")
    plt.colorbar()

    plt.subplot(222)
    plt.title("Transformada del filtro Gaussiano")
    freq_x_g = np.fft.fftfreq(spec_gauss.shape[0], d=1 / (2 * spec_gauss.max()))
    freq_y_g = np.fft.fftfreq(spec_gauss.shape[1], d=1 / (2 * spec_gauss.max()))
    plt.imshow(spec_gauss, cmap='hsv', extent=(freq_x_g.min(), freq_x_g.max(), freq_y_g.min(), freq_y_g.max()))
    plt.ylabel("u (frecuencias verticales)")
    plt.xlabel("v (frecuencias horizontales)")
    plt.colorbar()

    plt.subplot(223)
    plt.title("Transformada del filtro detector de bordes")
    freq_x_b = np.fft.fftfreq(spec_border.shape[0], d=1 / (2 * spec_border.max()))
    freq_y_b = np.fft.fftfreq(spec_border.shape[1], d=1 / (2 * spec_border.max()))
    plt.imshow(spec_border, cmap='hsv', extent=(freq_x_b.min(), freq_x_b.max(), freq_y_b.min(), freq_y_b.max()))
    plt.ylabel("u (frecuencias verticales)")
    plt.xlabel("v (frecuencias horizontales)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('transform_plot.png', bbox_inches='tight', dpi=100)

# TODO: Normalize results, so data can be compared.