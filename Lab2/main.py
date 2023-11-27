import cv2
import numpy as np
import kernels as k
import time

np.set_printoptions(threshold = np.inf)




def bw_method(image):
    height, width,_ = image.shape
    image1 = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]
            gray_value = int(0.36 * r + 0.53 * g + 0.11 * b)
            image1[y, x] = gray_value

    return image1




def filter(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(image, max(kernel_height//2, kernel_width//2), mode='constant', constant_values=0)

    filtered_image = np.zeros_like(image, dtype=float)

    for i in range(image_height):
        for j in range(image_width):
            """
            patch это подстрока которая имеет такой же размер как и ядро,
            'i: i + kernel_height' указывает диапазон строк для извлечения:
            начинается с текущего значения i и продолжается до i + kernel_height - 1.
            'j: j + kernel_width' указывает диапазон столбцов для извлечения
            начинается с текущего значения j и продолжается до j + kernel_height - 1.
            """
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            # print(np.column_stack((patch, kernel)))
            filtered_pixel = np.sum(patch * kernel)
            filtered_image[i, j] = filtered_pixel
            result_image = filtered_image
            # result_image = np.clip(filtered_image, 0, 255)
    return result_image

def scalling(filtered_image):
    min_val = np.min(filtered_image)
    max_val = np.max(filtered_image)
    range_val = max_val - min_val
    scaled_image = 255 * (filtered_image - min_val) / range_val
    result_image = scaled_image.astype(np.uint8)
    return result_image


kernel_inv = k.inversia()
kernel_gaus = k.gaussian_kernel(11, 1)
kernel_blur = k.blur_move_diagonal()
kernel_sharp = k.sharpness()
kernel_sobel = k.sobel_diag()
kernel_border = k.border()
kernel_relief = k.relief()

kernels = [kernel_inv, kernel_gaus, kernel_blur, kernel_sharp, kernel_sobel, kernel_border, kernel_relief]
kernel_names = ['kernel_inv', 'kernel_gaus', 'kernel_blur', 'kernel_sharp', 'kernel_sobel', 'kernel_border',
                'kernel_relief']

def download_all_kernels(img, img_name: str):
    for i, kernel in enumerate(kernels):
        result = filter(img, kernel)
        if i == 0:
            result = scalling(result) #for invercia
        cv2.imwrite(f'Images_output\convolved_{img_name}_{kernel_names[i]}.jpg', result)
        print(f'end {i}')

def download_all_kernels_cv(img, img_name: str):
    for i, kernel in enumerate(kernels):
        cv_image = cv2.filter2D(img, -1, kernel)
        if i == 0:
            cv2.imwrite(f'Images_output\convolved_{img_name}_{kernel_names[i]}_not_working_cv.jpg', cv_image)
            continue
        cv2.imwrite(f'Images_output\convolved_{img_name}_{kernel_names[i]}_cv.jpg', cv_image)
        print(f'cv end {i}')

def download_inversia_cv(img, img_name):
    inverted_image = cv2.bitwise_not(img)
    cv2.imwrite(f'Images_output\convolved_{img_name}_kernel_inv_cv.jpg', inverted_image)


def download_one_kernel(img, img_name: str, kernel, kernel_name):
    result = filter(img, kernel)
    cv2.imwrite(f'Images_output\convolved_image_{img_name}_{kernel_name}.jpg', result)
    print(f'end')


def download_shift_image(img, img_name):
    move_20down_and_10right = np.zeros((41, 41))
    move_20down_and_10right[0, 9] = 1
    shifted_image = filter(img, move_20down_and_10right)
    cv2.imwrite(f'Images_output\shifted_image_{img_name}.jpg', shifted_image)




"""Тесты на искуственных изображениях"""
# test_image = np.array([[0,255,0,0,255,0],
#                        [255,255,255,0,255,0],
#                        [0,255,0,0,255,255],
#                        [0,255,255,0,255,255],
#                        [255,0,255,0,0,255],
#                        [0,0,0,0,255,0]])
# test_image = np.zeros((101,101))
# test_image[0, 0] = 255
# test_image[0, 100] = 255
# test_image[100, 100] = 255
# test_image[100, 0] = 255
# test_kernel = np.array([[0,1,0],
#                         [0,0,0],
#                         [0,0,0]])

# Создаем массив 21x21, заполненный нулями

# move_20down_and_10right = np.zeros((41, 41))
# move_20down_and_10right[0, 9] = 1
# print(move_20down_and_10right)
#
# # test_image_shift = shift_20_bottom(shift_10_right(test_image))
# test_image_shift = filter(test_image, move_20down_and_10right)
# cv2.imwrite(f'Images_output/test_image.jpg', test_image)
# cv2.imwrite(f'Images_output/test_image_shift.jpg', test_image_shift)
