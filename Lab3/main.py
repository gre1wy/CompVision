import cv2
import numpy as np

image_read = cv2.imread(r'images\text.jpg', )
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
def erosion(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(image, max(kernel_height//2, kernel_width//2), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(image_height):
        for j in range(image_width):
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            filtered_image[i, j] = np.min(patch * kernel)

    return filtered_image

def dilation(image, kernel): #розширення
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(image, max(kernel_height//2, kernel_width//2), mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(image_height):
        for j in range(image_width):
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            filtered_image[i, j] = np.max(patch * kernel)

    return filtered_image
def closing(image, kernel):
    img1 = dilation(image, kernel)
    img2 = erosion(img1, kernel)
    return img2
def opening(image, kernel):
    img1 = erosion(image, kernel)
    img2 = dilation(img1, kernel)
    return img2

kernel = np.ones((2, 2), np.uint8)

test1 = erosion(image, kernel)
test2 = dilation(image, kernel)
test3 = closing(image, kernel)
test4 = opening(image, kernel)
# Convert the result back to uint8 before writing to an image file
# test1 = np.uint8(test1)
# test2 = np.uint8(test2)
cv2.imwrite(f'test1.jpg', test1)
cv2.imwrite(f'test2.jpg', test2)
cv2.imwrite(f'test3.jpg', test3)
cv2.imwrite(f'test4.jpg', test4)
