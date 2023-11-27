import main as m
import cv2
from kernels import my_kernel
""" Сhoose image """
img_name = ''
image_read = cv2.imread(fr'images/{img_name}.jpg')
image = m.bw_method(image_read)
cv2.imwrite(f'Images_output/{img_name}_bw.jpg', image)
m.download_all_kernels(image, img_name)
m.download_all_kernels_cv(image, img_name)
m.download_inversia_cv(image, img_name)
m.download_shift_image(image, img_name)

""" Download my kernel """
kernel1 = my_kernel()
m.download_one_kernel(image, img_name, kernel1, 'my_kernel') #looks like sobel but worse