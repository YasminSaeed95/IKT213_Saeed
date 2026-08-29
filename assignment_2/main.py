import cv2
import numpy as np

image = cv2.imread("iris-1.png")

# 1. Padding Task
def padding(image, border_width):
    return cv2.copyMakeBorder(
        image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_REFLECT
    )

padded_image = padding(image,100)
cv2.imwrite("padded_iris.png", padded_image)


# 2. Crop Task
def crop(image, x_0, x_1,  y_0, y_1):
    return image[y_0:y_1, x_0:x_1]

height, width = image.shape[:2]
x_0, x_1 = 200, width - 130
y_0, y_1 = 200, height - 130

cropped_img = crop(image, x_0, x_1, y_0, y_1)
cv2.imwrite("cropped_iris.png", cropped_img)


# 3. Resize Task
def resize(image, width, height):
    return cv2.resize(image, (width, height))

resized_img = resize(image, width=200, height=200)
cv2.imwrite("resized_iris.png", resized_img)


# 4. Copy Task
def copy(image, emptyPictureArray):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]

    return emptyPictureArray

# Manual Copy
height, width, channels = image.shape
emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)

copied_image = copy(image, emptyPictureArray)
cv2.imwrite("iris_copy.png", copied_image)


# 5. Greyscale Task
def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

greyscaled_image = greyscale(image)
cv2.imwrite("greyscaled_iris.png", greyscaled_image)

# 6. HSV Task (Hue: hvilken farge? , Saturation: hvor sterk/ren  er fargen? , Value: hvor lys/mørk er fargen)
def hsv (image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

hsv_image = hsv(image)
cv2.imwrite("hsv_iris.png", hsv_image)


# 7. Color Shifting Task (hue=50)
def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                emptyPictureArray[y, x, c] = np.clip(
                    int(image[y, x, c]) + hue,
                    0,
                    255
                )

    return emptyPictureArray

# Hue shifting
height, width, channels = image.shape

emptyPictureArray = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

shifted_image = hue_shifted(
    image,
    emptyPictureArray,
    50
)

cv2.imwrite("iris_hue_shifted.png", shifted_image)

# 8. Smoothing Task (ksize=(15,15))
def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

smoothed_image = smoothing(image)
cv2.imwrite("smoothed_iris.png", smoothed_image)

# 9. Rotation Task
def rotation (image, rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)

    return rotated_image

rotatated_180 = cv2.rotate(image, cv2.ROTATE_180)
cv2.imwrite("rotated_iris.png", rotatated_180)







