import cv2
import numpy as np
import kernels as k
import time

start = time.time() ## точка отсчета времени

np.set_printoptions(threshold = np.inf)
image = cv2.imread(r'images\Test.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('bw.jpg', image)
kernel = k.Rizkist()

#Тест на рандом np.array
#np.random.seed(42)
#random_array = np.random.randint(0, 256, size=(10, 10))
#print('random_array')
#print(random_array)

def filter2D(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(image, max(kernel_height//2, kernel_width//2), mode='constant', constant_values=0)
    #print('padded_image')
    #print(padded_image)
    #print('kernel')
    #print(kernel)
    filtered_image = np.zeros_like(image, dtype=float)
    #print('filtered_image')
    #print(filtered_image)

    for i in range(image_height):
        for j in range(image_width):
            # patch это подстрока которая имеет такой же размер как и ядро,
            # 'i: i + kernel_height' указывает диапазон строк для извлечения:
            # начинается с текущего значения i и продолжается до i + kernel_height - 1.
            # 'j: j + kernel_width' указывает диапазон столбцов для извлечения
            # начинается с текущего значения j и продолжается до j + kernel_height - 1.
            # например первый патч: [[   0.   -0.    0.]
            #                        [  -0.  510. -179.]
            #                        [   0. -210.    0.]]
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            #print(patch)
            filtered_pixel = np.sum(patch * kernel)
            filtered_image[i, j] = filtered_pixel

    #print(filtered_image)
    result_image = np.clip(filtered_image, 0, 255)
    print(result_image)
    return result_image
result = filter2D(image, kernel)
cv2.imwrite('convolved_image.jpg', result)
print('end')
end = time.time() - start ## собственно время работы программы

print(end) ## вывод времени