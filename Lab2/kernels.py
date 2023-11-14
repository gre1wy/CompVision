import numpy as np
def shift_kernel_10right():
    matrix = np.zeros((20, 20), dtype=np.float32)
    matrix[4, 9] = 1
    return matrix
def shift_kernel_20down():
    matrix = np.zeros((20, 20), dtype=np.float32)
    matrix[19, 9] = 1
    return matrix

def Inversia():
    return np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
        ], dtype=np.float32)

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)
gaussian_kernel_7x7 = gaussian_kernel(7, 0.8)

def blur_move_diagonal():
    return (1/7)*np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)

def sharpness():
    return np.array([
        [0, -1, 0,],
        [-1, 5, -1,],
        [0, -1, 0,],
        ], dtype=np.float32)

def sobel_diag():
    return np.array([
        [-2, -1, 0,],
        [-1, 0, 1,],
        [0, 1, 2,],
        ], dtype=np.float32)

def border():
    return np.array([
        [-1, -1, -1,],
        [-1, 8, -1,],
        [-1, -1, -1,],
        ], dtype=np.float32)

def relief():
    return np.array([
        [-2, -1, 0,],
        [-1, 1, 1,],
        [0, 1, 2,]
        ], dtype=np.float32)