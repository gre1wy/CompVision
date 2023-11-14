import cv2
import numpy as np
import kernels as k
import time

start = time.time()  # точка отсчета времени

np.set_printoptions(threshold = np.inf)

image_read = cv2.imread(r'images\.jpg')
def bw_method(image):
    height, width, _ = image.shape
    image1 = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]
            gray_value = int(0.36 * r + 0.53 * g + 0.11 * b)
            image1[y, x] = gray_value

    return image1


image = bw_method(image_read)


def filter(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(image, max(kernel_height//2, kernel_width//2), mode='constant', constant_values=0)
    # print('padded_image')
    # print(padded_image)
    # print('kernel')
    # print(kernel)
    filtered_image = np.zeros_like(image, dtype=float)
    # print('filtered_image')
    # print(filtered_image)

    for i in range(image_height):
        for j in range(image_width):
            # patch это подстрока b которая имеет такой же размер как и ядро,
            # 'i: i + kernel_height' указывает диапазон строк для извлечения:
            # начинается с текущего значения i и продолжается до i + kernel_height - 1.
            # 'j: j + kernel_width' указывает диапазон столбцов для извлечения
            # начинается с текущего значения j и продолжается до j + kernel_height - 1.
            # например первый патч: [[   0.   -0.    0.]
            #                        [  -0.  510. -179.]
            #                        [   0. -210.    0.]]
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            # print(patch)
            filtered_pixel = np.sum(patch * kernel)
            filtered_image[i, j] = filtered_pixel
            result_image = filtered_image
            #result_image = scalling(filtered_image) #для инверсии
            #result_image = np.clip(filtered_image, 0, 255)
    return result_image

def scalling(filtered_image):
    min_val = np.min(filtered_image)
    max_val = np.max(filtered_image)
    range_val = max_val - min_val
    scaled_image = 255 * (filtered_image - min_val) / range_val
    result_image = scaled_image.astype(np.uint8)
    return result_image


kernel0 = k.Inversia()
kernel1 = k.gaussian_kernel(7, 0.8)
kernel2 = k.blur_move_diagonal()
kernel3 = k.sharpness()
kernel4 = k.sobel_diag()
kernel5 = k.border()
kernel6 = k.relief()

kernels = [kernel0, kernel1, kernel2, kernel3, kernel4, kernel5, kernel6]

def download_all_kernels():
    for i, kernel in enumerate(kernels):
        result = filter(image, kernel)
        if i == 0:
            result = scalling(result) #for invercia
        cv2.imwrite(f'convolved_image{i}.jpg', result)
        print(f'end {i}')

def download_all_kernels_cv():
    for i, kernel in enumerate(kernels):
        cv_image = cv2.filter2D(image, -1, kernel)
        cv2.imwrite(f'convolved_image_cv{i}.jpg', cv_image)
        print(f'cv end {i}')

def download_one_kernel():
    result = filter(image, kernel1) #change number
    cv2.imwrite(f'convolved_image{1}.jpg', result) #change number
    print(f'end')

download_all_kernels()
download_all_kernels_cv()
# download_one_kernel()
def download_shift_image():
    kernels1 = k.shift_kernel_10right()
    kernels2 = k.shift_kernel_20down()
    filtered_image1 = filter(image, kernels1)
    filtered_image2 = filter(filtered_image1, kernels2)
    cv2.imwrite('shift_10_20.jpg', filtered_image2.astype(np.uint8))

download_shift_image()


end = time.time() - start  # собственно время работы программы

print(end)  # вывод времени
