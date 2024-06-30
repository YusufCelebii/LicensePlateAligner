import cv2 as cv
import numpy as np
import os

def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray, 5)
    high_contrast = cv.convertScaleAbs(blurred, alpha=2, beta=-50)
    _, thresholded = cv.threshold(high_contrast, 200, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresholded, None, iterations=1)
    return dilated


def find_largest_contour(dilated):
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv.contourArea)
    return None


def simplify_contour(contour):
    hull = cv.convexHull(contour)
    epsilon = 0.045 * cv.arcLength(hull, True)
    simplified_hull = cv.approxPolyDP(hull, epsilon, True)
    return simplified_hull


def mark_corners(img, contour):
    corners = cv.goodFeaturesToTrack(cv.cvtColor(img, cv.COLOR_BGR2GRAY), maxCorners=4, qualityLevel=0.01,
                                     minDistance=10)
    if corners is not None:
        corners = np.intp(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)


def draw_contours(img, contour):
    hull_img = np.zeros_like(img)
    cv.drawContours(hull_img, [contour], -1, (0, 0, 255), 5)
    mark_corners(hull_img, hull_img)
    return hull_img


def draw_and_save_hull_on_original(img, contour, output_path):
    cv.drawContours(img, [contour], -1, (0, 0, 255), 5)
    cv.imwrite(output_path, img)


def expand_corners(corners, expansion=60):
    expanded_corners = []
    for corner in corners:
        x, y = corner.ravel()
        expanded_corners.append([x - expansion, y - expansion])
        expanded_corners.append([x + expansion, y - expansion])
        expanded_corners.append([x + expansion, y + expansion])
        expanded_corners.append([x - expansion, y + expansion])
    return np.array(expanded_corners, dtype=np.float32)


def auto_perspective_transform(corners, original):
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    expanded_corners = expand_corners(corners)
    src = order_points(expanded_corners)


    dst_width, dst_height = 500, 100
    dst = np.array([
        [0, 0],
        [dst_width, 0],
        [dst_width, dst_height],
        [0, dst_height]
    ], dtype=np.float32)

    matrix = cv.getPerspectiveTransform(src, dst)
    transformed = cv.warpPerspective(original, matrix, (dst_width, dst_height))

    return transformed


def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    convex_dir = os.path.join(output_dir, "convex")
    if not os.path.exists(convex_dir):
        os.makedirs(convex_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv.imread(img_path)

            preprocessed = preprocess_image(img)
            largest_contour = find_largest_contour(preprocessed)

            if largest_contour is not None:
                simplified_hull = simplify_contour(largest_contour)
                hull_img = draw_contours(img.copy(), simplified_hull)
                corners = cv.goodFeaturesToTrack(cv.cvtColor(hull_img, cv.COLOR_BGR2GRAY), maxCorners=4,
                                                 qualityLevel=0.01, minDistance=10)
                if corners is not None:
                    corners = np.intp(corners)
                    transformed_img = auto_perspective_transform(corners, img)
                    output_path = os.path.join(output_dir, filename)
                    cv.imwrite(output_path, transformed_img)


                    convex_output_path = os.path.join(convex_dir, filename)
                    draw_and_save_hull_on_original(img.copy(), simplified_hull, convex_output_path)
                else:
                    print(f"{filename} için köşe noktası bulunamadı")
            else:
                print(f"{filename} için kontur bulunamadı")


input_dir = "sample"
output_dir = "results"
process_images(input_dir, output_dir)



