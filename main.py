import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray, 5)
    high_contrast = cv.convertScaleAbs(blurred, alpha=2, beta=-50)
    _, thresholded = cv.threshold(high_contrast, 200, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresholded, None, iterations=1)
    return dilated

def find_largest_contour(edges):
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_cnt = None
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    return max_cnt

def correct_perspective(image, contour):
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    box = np.intp(box)
    box = sorted(box, key=lambda x: (x[0], x[1]))

    if box[1][1] > box[0][1]:
        top_left, bottom_left = box[0], box[1]
    else:
        top_left, bottom_left = box[1], box[0]

    if box[3][1] > box[2][1]:
        top_right, bottom_right = box[2], box[3]
    else:
        top_right, bottom_right = box[3], box[2]

    width = 400
    height = 100

    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(image, M, (width, height))
    return warped


img_path = r"frame_00452_cropped.jpg"
img = cv.imread(img_path)

img1 = preprocess_image(img)
contour = find_largest_contour(img1)


img_with_contour = img.copy()
rect = cv.minAreaRect(contour)
box = cv.boxPoints(rect)
box = np.intp(box)
cv.drawContours(img_with_contour, [box], 0, (0, 0, 255), 3)


warped_img = correct_perspective(img, contour)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(img_with_contour, cv.COLOR_BGR2RGB))
plt.title("Contour Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(warped_img, cv.COLOR_BGR2RGB))
plt.title("Warped Image")
plt.axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Görseller arası boşluğu azalt
plt.show()
