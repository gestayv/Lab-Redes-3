import redes3 as r3

def main_r3(image_name, custom_kernel):
    data = r3.normalize_img(r3.open_img(image_name), 0, 1)
    data_gauss = r3.normalize_img(r3.filter_gauss(data), 0, 1)
    data_border = r3.normalize_img(r3.filter_border(data), 0, 1)
    data_custom = r3.normalize_img(r3.filter_img(data, custom_kernel), 0, 1)

    r3.save_img("filter_gauss", data_gauss)
    r3.save_img("filter_border", data_border)
    r3.save_img("filter_custom", data_custom)

    r3.plot_aside(data, "Transformada de la imagen original", "original_transform")
    r3.plot_aside(data_gauss, "Transformada del filtro Gaussiano", "gauss_transform")
    r3.plot_aside(data_border, "Transformada del filtro detector de bordes", "border_transform")
    r3.fourier_transform(data, data_gauss, data_border)


kernel = [[-1, -1, -1],
          [-1, 0, -1],
          [-1, -1, -1]]

main_r3("leena512.bmp", kernel)
