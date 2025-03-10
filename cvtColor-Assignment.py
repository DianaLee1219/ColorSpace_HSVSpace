# Import modules
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dataPath import DATA_PATH
%matplotlib inline

import matplotlib
matplotlib.rcParams['figure.figsize'] = (18.0, 12.0)
matplotlib.rcParams['image.interpolation'] = 'bilinear'

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

# Image loading and display
img = cv2.imread(DATA_PATH+"images/sample.jpg")
plt.imshow(img[:,:,::-1])
plt.show()

def convertBGRtoGray(image):
    channel_R = image[:,:,2]
    channel_G = image[:,:,1]
    channel_B = image[:,:,0]
    
    channel_Gray = 0.299*channel_R + 0.587*channel_G + 0.114*channel_B
    return channel_Gray

def convertBGRtoHSV(image):
    height, width, _ = image.shape  # Correctly unpack shape
    
    # Extract channels
    channel_R = image[:,:,2].astype(float)  # Convert to float for calculations
    channel_G = image[:,:,1].astype(float)
    channel_B = image[:,:,0].astype(float)

    # Initialize output arrays
    H = np.zeros((height, width), dtype=np.float32)
    S = np.zeros((height, width), dtype=np.float32)
    V = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            R = channel_R[i, j]
            G = channel_G[i, j]
            B = channel_B[i, j]

            # Compute Value (Brightness)
            channel_V = max(R, G, B)
            min_RGB = min(R, G, B)
            delta = channel_V - min_RGB  # Difference

            # Compute Saturation
            if channel_V == 0:
                channel_S = 0
            else:
                channel_S = delta / channel_V
            
            # Compute Hue
            if delta == 0:
                channel_H = 0  # Undefined hue for grayscale colors
            elif channel_V == R:
                channel_H = 60 * ((G - B) / delta )
            elif channel_V == G:
                channel_H = 60 * ((B - R) / delta + 2)
            else:  # channel_V == B
                channel_H = 60 * ((R - G) / delta + 4)

            if channel_H < 0:
                channel_H += 360

            # Convert to OpenCV scale (H: 0-180, S & V: 0-255)
            H[i, j] = channel_H / 2  # OpenCV scales Hue to [0, 180]
            S[i, j] = channel_S * 255
            V[i, j] = channel_V  # Already in [0, 255]

    return H.astype(np.uint8), S.astype(np.uint8), V.astype(np.uint8)

gray = convertBGRtoGray(img)
gray_cv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Convert using custom function
H, S, V = convertBGRtoHSV(img)
hsv = np.stack([H, S, V], axis=-1)  # Convert tuple to NumPy array
hsv_cv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

plt.figure(figsize=(18,12))
plt.subplot(1,3,1)
plt.title("Result from custom function")
plt.imshow(gray,cmap="gray")
plt.subplot(1,3,2)
plt.title("Result from OpenCV function")
plt.imshow(gray_cv,cmap="gray")
plt.subplot(1,3,3)
plt.title("Difference")
plt.imshow(np.abs(gray-gray_cv),cmap="gray")
plt.show()

plt.subplot(1,3,1)
plt.title("Result from custom function")
plt.imshow(hsv[:,:,::-1])
plt.subplot(1,3,2)
plt.title("Result from OpenCV function")
plt.imshow(hsv_cv[:,:,::-1])
plt.subplot(1,3,3)
plt.title("Difference")
plt.imshow(np.abs(hsv-hsv_cv)[:,:,::-1])
plt.show()
