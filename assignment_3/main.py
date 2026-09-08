import cv2
import numpy as np

image = cv2.imread("lambo.png")
image2 =cv2.imread("shapes-1.png")
template = cv2.imread("shapes_template.jpg")

#Sobel Edge Detection Task
def sobel_edge_detection(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)   #(3,3) er størrelsen på området som brukes for å beregne blur.
    return  cv2.Sobel(blurred_image, cv2.CV_64F, 1, 1, ksize=1)   #Sobel prøver å finne store endringer mellom nabopiksler (dette kan være en kant).

sobel = sobel_edge_detection(image)
cv2.imwrite("sobel_lambo.png", sobel)


#Canny Edge Detection Task
def canny_edge_detection(image, threshold_1, threshold_2):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.Canny(blurred_image, threshold_1, threshold_2)

canny = canny_edge_detection(image,50,50)
cv2.imwrite("canny_lambo.png", canny)


#Template Match Task
def template_match (image2, template):

    # 1.Convert both image2 and template to greyscale
    gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 2.Find matches
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)  #“result” contains a lot of numbers. These numbers indicate how well the template matches each area.

    # 3.Threshold
    threshold = 0.9
    locations = np.where( result >= threshold)

    # 4. Get template size
    template_height, template_width = gray_template.shape

    # 5. Draw rectangle around every match
    for pt in zip(*locations[::-1]):
        cv2.rectangle(image2, pt, (pt[0] + template_width, pt[1] + template_height), (0,0,255), 2)

    return image2

match = template_match(image2, template)
cv2.imwrite('match.png',match)


#Resizing Task
def resize(image, scale_factor: int, up_or_down: str ): #scale factor means how many times we should go up/down
    if up_or_down.lower() == "up":
        for i in range (scale_factor):
            image= cv2.pyrUp(image)


    elif up_or_down.lower() == "down":
        for i in range (scale_factor):
            image= cv2.pyrDown(image)

    return image

resized_image = resize(image, scale_factor=2, up_or_down="up")
cv2.imwrite('resized.png',resized_image)
print("Original:", image.shape)
print("Resized:", resized_image.shape)