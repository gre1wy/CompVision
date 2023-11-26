import main as m
import numpy as np
import cv2
np.set_printoptions(threshold = np.inf)



"""Обработка тестовой картинки"""

# test_image = np.array([[0,255,0,0,255,0],
#                        [255,255,255,0,255,0],
#                        [0,255,0,0,255,255],
#                        [0,255,255,0,255,255],
#                        [255,0,255,0,0,255],
#                        [0,0,0,0,255,0]])
# test_kernel = np.array([[0,255,0],
#                         ['sp0',255,0],
#                         [0,255,0]])
#
#
# test1 = m.dilation(test_image, test_kernel)
# test2 = m.erosian(test_image, test_kernel)
# test3 = m.closing(test_image, test_kernel)
# test4 = m.opening(test_image, test_kernel)
# test5 = m.limits(test_image, test_kernel)
# cv2.imwrite(f'Images_output/{"0original_image_"+"test_image"}monky.jpg', test_image)
# cv2.imwrite(f'Images_output/{"1dilation_image_"+"test_image"}monky.jpg', test1)
# cv2.imwrite(f'Images_output/{"2erosion_image_"+"test_image"}monky.jpg', test2)
# cv2.imwrite(f'Images_output/{"3closing_image_"+"test_image"}monky.jpg', test3)
# cv2.imwrite(f'Images_output/{"4opening_image_"+"test_image"}monky.jpg', test4)
# cv2.imwrite(f'Images_output/{"5limits_image_"+"test_image"}monky.jpg', test5)

"""Обработка изображения"""

name = "apple"
# name = 'Text'
image_read = cv2.imread(f'images/{name}.jpg')

image1 = m.bw_method(image_read)
image = m.binarization_on_bw_image(image1)

kernel = np.array([[0,255,0],
                   [0,255,0],
                   [0,255,0]])

dilation1 = m.dilation(image, kernel)
erosion2 = m.erosian(image, kernel)
closing3 = m.closing(image, kernel)
opening4 = m.opening(image, kernel)
limits5 = m.limits(image, kernel)

cv2.imwrite(f'Images_output/{"0original_image_"+name}.jpg', image)
cv2.imwrite(f'Images_output/{"1dilation_image_"+name}.jpg', dilation1)
cv2.imwrite(f'Images_output/{"2erosion_image_"+name}.jpg', erosion2)
cv2.imwrite(f'Images_output/{"3closing_image_"+name}.jpg', closing3)
cv2.imwrite(f'Images_output/{"4opening_image_"+name}.jpg', opening4)
cv2.imwrite(f'Images_output/{"5limits_image_"+name}.jpg', limits5)

