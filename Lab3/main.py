import cv2
import numpy as np

def bw_method(image):
    height, width, _ = image.shape
    image1 = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            r, g, b = image[y, x]
            gray_value = int(0.36 * r + 0.53 * g + 0.11 * b)
            image1[y, x] = gray_value

    return image1

def binarization_on_bw_image(image):
    height, width = image.shape
    image1 = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if image[y, x] >= 128:
                image1[y, x] = 255
            else:
                image[y, x] = 0
    return image1


def process_special_point(matrix):
    # Создаем копию матрицы
    matrix_copy = matrix.copy()

    # Найти координаты элемента, начинающегося на 'sp'
    coordinates = np.argwhere(np.char.startswith(matrix_copy.astype(str), 'sp'))

    if len(coordinates) == 0:
        # print("Элементы, начинающиеся на 'sp', не найдены.")
        return matrix_copy, [0, 0]

    # Извлечь значения после 'sp', заменить в матрице
    for coord in coordinates:
        value = matrix_copy[coord[0], coord[1]][2:]  # Получаем значение после 'sp'
        matrix_copy[coord[0], coord[1]] = value

    # Преобразовать копию матрицы в int
    matrix_copy = matrix_copy.astype(int)

    # Найти центр матрицы
    center = np.array(matrix_copy.shape) // 2

    # Найти разницу между координатами и центром матрицы
    differences = coordinates - center

    return matrix_copy, differences[0].tolist()


def dilation(image, kernel):
    result_kernel, differences = process_special_point(kernel)
    # print(differences)
    offset_y = differences[0]
    offset_x = differences[1]

    result_kernel = result_kernel.astype(int) #не знаю почему без этого возвращает тип u32

    image_height, image_width = image.shape
    kernel_height, kernel_width = result_kernel.shape

    # Создаем изображение как входное, только с нулями на границах взависимости от ядра
    padded_image = np.pad(image, max(kernel_height//2, kernel_width//2), mode='constant', constant_values=0)
    # Создаем изображение размером как входное, только с нулями
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    # Находим координаты белых пикселей в ядре что бы потом сравнивать с белыми пикселями в патче
    where_white_pixel_kernel = np.where(result_kernel != 0)
    check_kernel = list(zip(where_white_pixel_kernel[0], where_white_pixel_kernel[1]))
    # t = 0

    for i in range(image_height):
        for j in range(image_width):

            if i + offset_y < 0 or j + offset_x < 0:
                # print('Выходит за границу изображения')
                continue

            # Создаем патч из изображения для каждой координаты
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]

            # print(np.column_stack((patch, result_kernel)))

            # Находим координаты белых точек после прикладывания патча
            where_white_pixel_patch = np.where(result_kernel * patch != 0)
            check_patch = list(zip(where_white_pixel_patch[0], where_white_pixel_patch[1]))
            try:
                # Проверяем есть ли хоть один одинаковый белый пиксель
                if any(elem in check_patch for elem in check_kernel):
                    filtered_image[i+offset_y, j+offset_x] = 255
                    # print([i+offset_y, j+offset_x])
                    # print('hit')
                else:
                    filtered_image[i+offset_y, j+offset_x] = 0
                    # print('miss')
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                filtered_image[i, j] = image[i, j]
                pass
            # t +=1
            # print(f"image {t}")
            # print(filtered_image)
    return filtered_image


def erosian(image, kernel):
    result_kernel, differences = process_special_point(kernel)
    offset_y = differences[0]
    offset_x = differences[1]
    # print(differences)
    result_kernel = result_kernel.astype(int)

    image_height, image_width = image.shape
    kernel_height, kernel_width = result_kernel.shape

    padded_image = np.pad(image, max(kernel_height//2, kernel_width//2), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    where_white_pixel_kernel = np.where(result_kernel != 0)
    check_kernel = list(zip(where_white_pixel_kernel[0], where_white_pixel_kernel[1]))

    # t = 0

    for i in range(image_height):
        for j in range(image_width):
            if i + offset_y < 0 or j + offset_x < 0:
                # print('Выходит за границу изображения')
                continue
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            where_white_pixel_patch = np.where(result_kernel * patch != 0)
            check_patch = list(zip(where_white_pixel_patch[0], where_white_pixel_patch[1]))
            # print(np.column_stack((patch, result_kernel)))
            try:
                if all(elem in check_patch for elem in check_kernel):
                    filtered_image[i+offset_y, j+offset_x] = 255
                    # print([i+offset_y, j+offset_x])
                    # print('fit')
                else:
                    filtered_image[i+offset_y, j+offset_x] = 0
                    # print([i + offset_y, j + offset_x])
                    # print('miss')
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                filtered_image[i, j] = 0
                pass
            # t +=1
            # print(f"image {t}")
            # print(filtered_image)

    return filtered_image
def closing(image, kernel):
    img1 = dilation(image, kernel)
    img2 = erosian(img1, kernel)
    return img2
def opening(image, kernel):
    img1 = erosian(image, kernel)
    img2 = dilation(img1, kernel)
    return img2
def limits(image, kernel):
    return image ^ erosian(image, kernel)

