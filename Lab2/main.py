import cv2
import numpy as np
import matplotlib.pyplot as plt
#image = cv2.imread(r'images/.jpg')
image = cv2.imread(r'images\.jpg', cv2.IMREAD_GRAYSCALE)

# Define a 3x3 kernel (example: identity kernel)
#kernel = np.array([[0, 0, 0],
                  #[0, 0, 1],
                  #[0, 0, 0]])

kernel = (1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, 36, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]])
# Get the dimensions of the image and kernel
image_height, image_width = image.shape
kernel_height, kernel_width = kernel.shape

# Create an output image (result) with the same dimensions
result = np.zeros((image_height, image_width), dtype=np.uint8)

# Perform convolution
for y in range(image_height):
    for x in range(image_width):
        for ky in range(kernel_height):
            for kx in range(kernel_width):
                # Calculate the coordinates in the kernel
                k_x = kernel_width - 1 - kx
                k_y = kernel_height - 1 - ky

                # Calculate the coordinates in the image
                img_x = x - kernel_width // 2 + k_x
                img_y = y - kernel_height // 2 + k_y

                # Check if the coordinates are within the image bounds
                if 0 <= img_x < image_width and 0 <= img_y < image_height:
                    result[y, x] += image[img_y, img_x] * kernel[k_y, k_x]

# Save the result
cv2.imwrite('convolved_image.jpg', result)
print(result.shape)