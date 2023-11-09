import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

Original_name = "cat.JPG" #Change image name
BWImage_name = "B/W_"+Original_name
Mask_name = "Mask_"+Original_name
Cut_image = "Cut_"+Original_name

input_image = cv2.imread(Original_name)

# Get the dimensions of the image
height, width, _ = input_image.shape

# Create an empty grayscale image of the same dimensions
bw_image = np.zeros((height, width), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        r, g, b = input_image[y, x]
        gray_value = int(0.36 * r + 0.53 * g + 0.11 * b)
        bw_image[y, x] = gray_value

plt.imshow(bw_image, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')  # Hide axis
plt.show()

hist = cv2.calcHist([bw_image], [0], None, [256], [0, 256])

smoothed_hist = cv2.GaussianBlur(hist, (5, 5), 0)
smoothed_hist[0] = 0  # Set the left boundary to zero
smoothed_hist[255] = 0  # Set the right boundary to zero
plt.plot(smoothed_hist)
peaks, _ = find_peaks(smoothed_hist.flatten(), height=0)
print(peaks)
sorted_peaks = sorted(peaks, key=lambda x: smoothed_hist[x], reverse=True)

# Take the two highest peaks
highest_peak_1 = sorted_peaks[0]
highest_peak_2 = sorted_peaks[1]

print(f"First highest peak: {highest_peak_1}, Value: {smoothed_hist[highest_peak_1]}")
print(f"Second highest peak: {highest_peak_2}, Value: {smoothed_hist[highest_peak_2]}")

min_value = float('inf')
min_index = -1

for i in range(highest_peak_2, highest_peak_1 + 1):
    if hist[i] < min_value:
        min_value = hist[i]
        min_index = i

print(f"Minimum value: {min_value} at index: {min_index}")

# Відобразіть гістограму
plt.figure(figsize=(8, 6))
plt.title('Гістограма відтінків сірого (Grayscale Histogram)')
plt.xlabel('Відтінки сірого (Gray Level)')
plt.ylabel('Кількість пікселів')
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

threshold = min_index #Початковий поріг

#Застосовуємо порогову бінаризацію
Mask = np.where(bw_image < threshold, 1, 0)
#
object_cut = cv2.bitwise_and(input_image, input_image, mask=Mask.astype(np.uint8))

plt.imsave(Mask_name, Mask, cmap='gray')
plt.imshow(Mask, cmap='gray')
plt.title("Mask")
plt.axis('off')
plt.show()

cv2.imwrite(Cut_image, object_cut)  # Зберегти вирізаний об'єкт
cv2.imshow("Object Cut", object_cut)
cv2.waitKey(0)
cv2.destroyAllWindows()

