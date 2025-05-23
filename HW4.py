import cv2
import numpy as np
from scipy.spatial.distance import cdist
import math

np.random.seed(3)
img_path = "sunflowers.jpg"
img = cv2.imread(img_path)
# img = cv2.resize(img, (640, 640))
height, width, _ = img.shape
pixel_data = img.reshape(-1, 3)
original_bits = height * width * 3 * 8

K = 32
centers = pixel_data[np.random.choice(pixel_data.shape[0], K, replace=False)]
labels = np.random.randint(0, K, len(pixel_data))
iter = 0

while True:
    iter += 1
    distances = cdist(pixel_data, centers)
    nearest_center = np.argmin(distances, axis=1)
    if not np.array_equal(labels, nearest_center):
        labels = nearest_center
    else:
        break
    new_centers = []
    for k in range(K):
        same_label_index = np.where(labels == k)[0]
        if len(same_label_index) > 0:
            mean_point = pixel_data[same_label_index]
            new_centers.append(np.mean(mean_point, axis=0))
        else:
            new_centers.append(centers[k])
    centers = np.array(new_centers)

compressed_img = centers[labels].reshape(height, width , _).astype(np.uint8)

bits_per_pixel = math.log2(K)
bits_for_lookup = K * 24
compressed_bits = height * width * bits_per_pixel + bits_for_lookup
print(f"Số lần lặp: {iter} lần")
print(f"Số bit của ảnh gốc: {original_bits} bits")
print(f"Số bit của ảnh đã nén: {compressed_bits:.0f} bits")

puzzle_img = cv2.hconcat([img, compressed_img])
cv2.imshow("Image Compression", puzzle_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

