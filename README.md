# Color Space Conversion: BGR to Grayscale and HSV

## Introduction

In this assignment, we implemented functions to convert an image from the BGR color space to both Grayscale and HSV formats. Color space conversion is a crucial process in computer vision, 
enabling different representations of an image for various applications like object detection, segmentation, and color-based filtering.

## Theory of Color Spaces

A color space is a specific organization of colors, allowing consistent representation and reproduction across different devices. 
The **BGR (Blue, Green, Red)** color model is commonly used in OpenCV instead of the more familiar RGB order. Two other essential color representations are **Grayscale** and **HSV (Hue, Saturation, Value)**:

**Grayscale**: Represents an image with intensity values ranging from black (0) to white (255), reducing computational complexity.

**HSV (Hue, Saturation, Value)**: A more perceptually intuitive model where:

**Hue (H)**: Represents color type (0°-360° but scaled to 0-180 in OpenCV for 8-bit images).

**Saturation (S)**: Represents color purity, ranging from 0 (gray) to 255 (full color).

**Value (V)**: Represents brightness, with 0 being black and 255 being the brightest.

## Conversion Equations

1. BGR to Grayscale

Grayscale conversion is performed using a weighted sum of the BGR channels to reflect human perception:

![image](https://github.com/user-attachments/assets/a26a97a3-573a-48eb-b7f3-ca8697624cc6)


2. BGR to HSV

The transformation follows these steps:

Compute Value (V) as the maximum of R, G, and B:


Compute Saturation (S) as:


Compute Hue (H) based on which channel has the maximum value:

![image](https://github.com/user-attachments/assets/9d474ef7-ed82-464e-8349-638f2b8c43dd)


If H < 0, then H = H + 360.

Scaling for 8-bit images: ![image](https://github.com/user-attachments/assets/6c89ed95-9704-49c7-affc-b021926a96f1)


## Implementation & Results

We implemented the function convertBGRtoHSV(image) in Python using NumPy. The function iterates through each pixel, applies the conversion formulae, and returns the HSV image. The results were compared with OpenCV’s built-in cv2.cvtColor(image, cv2.COLOR_BGR2HSV), showing minimal difference, validating the correctness of our implementation. The function was then tested using Matplotlib to visualize the original and converted images.


![image](https://github.com/user-attachments/assets/0dbba96b-3b7c-4f0a-8557-3262e3812d85)

![image](https://github.com/user-attachments/assets/29740efc-e2f6-46f2-ba85-cd559e5dcb4c)

## Conclusion

This assignment provided hands-on experience in implementing color space transformations, reinforcing the theoretical knowledge behind image processing. The correct implementation of these conversions is crucial for various computer vision applications, enabling better feature extraction and analysis in tasks such as object recognition and segmentation.
