import redes3 as r3

def main_r3(image_name, custom_kernel):
    data = r3.normalize_img(r3.open_img(image_name))
    data_gauss = r3.filter_gauss(data)
    data_border = r3.filter_border(data)
    data_custom = r3.filter_img(data, custom_kernel)

    r3.save_img("filter_gauss", data_gauss)
    r3.save_img("filter_border", data_border)
    r3.save_img("filter_custom", data_custom)

    r3.fourier_transform(data, data_gauss, data_custom)


kernel = [[-1, -1, -1],
          [-1, 0, -1],
          [-1, -1, -1]]

main_r3("leena512.bmp", kernel)
