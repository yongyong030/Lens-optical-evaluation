import cv2
import numpy as np
import time

# File path name and image count
file_path = 'C:/Users/WTA/Documents/iwta/focusvarisample/'
num_images = 3

# Load the images
imgs = []
for i in range(num_images):
    img = cv2.imread(file_path + '{:03d}.bmp'.format(i))
    imgs.append(img)

def mma_img(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Filter the grayscale image with Gaussian Blur 1.5
    blur1 = cv2.GaussianBlur(gray, (0, 0), 1.5)

    # Subtract 1 throughout the image and take the absolute value
    absdiff1 = cv2.absdiff(gray, np.uint8(blur1))
    absdiff1 = cv2.convertScaleAbs(absdiff1)

    # Find the maximum value and its index
    max_val, max_idx = cv2.minMaxLoc(absdiff1)

    return max_idx[1]

# Synthesize focus stacking image
start_time = time.time()

stacked_image = np.zeros_like(imgs[0])
for i in range(stacked_image.shape[0]):
    for j in range(stacked_image.shape[1]):
        mma_idx = mma_img(imgs[i])
        stacked_image[i, j] = imgs[mma_idx][i, j]

elapsed_time = time.time() - start_time

# Display the synthesized focus stacking image
cv2.imshow("Synthesized Focus Stacking Image", stacked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Elapsed Time: {:.2f} seconds".format(elapsed_time))