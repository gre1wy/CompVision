import cv2
import numpy as np
np.set_printoptions(threshold = np.inf)
name = "apple"
image_read = cv2.imread(f'images/{name}.jpg')
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
def dilation(image, kernel, offset_y=0, offset_x=0):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(image, max(kernel_height//2, kernel_width//2), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    where_white_pixel_kernel = np.where(kernel != 0)
    check_kernel = list(zip(where_white_pixel_kernel[0], where_white_pixel_kernel[1]))
    # t = 0

    for i in range(image_height):
        for j in range(image_width):
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            # print(np.column_stack((patch, kernel)))
            where_white_pixel_patch = np.where(kernel * patch != 0)
            check_patch = list(zip(where_white_pixel_patch[0], where_white_pixel_patch[1]))
            try:
                if any(elem in check_patch for elem in check_kernel):
                    filtered_image[i+offset_y, j+offset_x] = 255
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

def erosian(image, kernel, offset_y=0, offset_x=0):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(image, max(kernel_height//2, kernel_width//2), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image, dtype=np.uint8)
    where_white_pixel_kernel = np.where(kernel != 0)
    check_kernel = list(zip(where_white_pixel_kernel[0], where_white_pixel_kernel[1]))
    for i in range(image_height):
        for j in range(image_width):
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            where_white_pixel_patch = np.where(kernel * patch != 0)
            check_patch = list(zip(where_white_pixel_patch[0], where_white_pixel_patch[1]))
            #print(check_kernel, check_patch)
            try:
                if all(elem in check_patch for elem in check_kernel):
                    filtered_image[i+offset_y, j+offset_x] = 255
                else:
                    filtered_image[i+offset_y, j+offset_x] = 0
            except Exception as e:
                print(f"Произошла ошибка: {e}")
                filtered_image[i, j] = 0
                pass

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

# test_image = np.array([[0,255,0,0,255,0],
#                        [255,255,255,0,255,0],
#                        [0,255,0,0,255,255],
#                        [0,255,255,0,255,255],
#                        [255,0,255,0,0,255],
#                        [0,0,0,0,255,0]])
# test_kernel = np.array([[0,255,0],
#                         [0,255,0],
#                         [0,255,0]])
#
# test1 = dilation(test_image, test_kernel)
# test2 = erosian(test_image, test_kernel)
# test3 = closing(test_image, test_kernel)
# test4 = opening(test_image, test_kernel)
# test5 = limits(test_image, test_kernel)
# cv2.imwrite(f'Images_output/{"0original_image_"+"test_image"}.jpg', test_image)
# cv2.imwrite(f'Images_output/{"1dilation_image_"+"test_image"}.jpg', test1)
# cv2.imwrite(f'Images_output/{"2erosion_image_"+"test_image"}.jpg', test2)
# cv2.imwrite(f'Images_output/{"3closing_image_"+"test_image"}.jpg', test3)
# cv2.imwrite(f'Images_output/{"4opening_image_"+"test_image"}.jpg', test4)
# cv2.imwrite(f'Images_output/{"5limits_image_"+"test_image"}.jpg', test5)


image1 = bw_method(image_read)
image = binarization_on_bw_image(image1)
kernel = np.array([[0,255,0],
                   [0,255,0],
                   [0,255,0]])
dilation1 = dilation(image, kernel)
erosion2 = erosian(image, kernel)
closing3 = closing(image, kernel)
opening4 = opening(image, kernel)
limits5 = limits(image, kernel)

cv2.imwrite(f'Images_output/{"0original_image_"+name}.jpg', image)
cv2.imwrite(f'Images_output/{"1dilation_image_"+name}.jpg', dilation1)
cv2.imwrite(f'Images_output/{"2erosion_image_"+name}.jpg', erosion2)
cv2.imwrite(f'Images_output/{"3closing_image_"+name}.jpg', closing3)
cv2.imwrite(f'Images_output/{"4opening_image_"+name}.jpg', opening4)
cv2.imwrite(f'Images_output/{"5limits_image_"+name}.jpg', limits5)
