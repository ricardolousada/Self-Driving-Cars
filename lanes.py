import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny_transformation(image):
    """
    :param image: Our original image
    :return: an image after applying the canny transformation
    """
    # create a gray scale image to speed up the process
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gaussian filter - Reduce noise to better identify edges
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    # Canny method short change of gradients (small or large derivatives) - valores a baixo do treahold ficam a preto
    canny = cv2.Canny(gray, 50, 150)
    return canny


def region_of_interest(image):
    """
    :param : an image to apply a mask
    :return: an image with a mask applied - this will be our region of interest
    """
    height = image.shape[0]
    width = image.shape[1]
    mask = np.zeros_like(image)
    triangle = np.array([[
        (200, height),
        (550, 250),
        (1100, height),]],np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    """
    :param image: Our original image
    :param lines: a two-dimensional array with the left and right line coordinates
    :return: a image with the detected lines
    """
    image_with_lines = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return image_with_lines


def make_coordinates(image, line_parameters):
    """
    :param image:Our image with the lanes
    :param lines: a tuple with a slope and an intercept
    :return: a array with the coordinates of the left and right lines
    """
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1,y1,x2,y2]]

def average_slop_intercept(image, lines):
    """
    :param image: the image with the lanes
    :param lines: the detected lines
    :return: a array with only two lines, containing the left and right line coordinates
    """
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            # print(parameters)
            slope = fit[0]
            intercept = fit[1]
            # lines on the left
            if slope < 0:
                global left_line
                left_fit.append((slope, intercept))
                left_fit_average = np.average(left_fit, axis=0)
                left_line = make_coordinates(image, left_fit_average)
            else:
                right_fit.append((slope, intercept))
    right_fit_average = np.average(right_fit, axis=0)
    right_line = make_coordinates(image, right_fit_average)
    if left_line is not None:
        return [left_line,right_line]
    else:
        return [right_line]



"""
input_image = cv2.imread('.\image\\test_image.jpg')
# Transform the image in a Matrix of pixels
lane_image = np.copy(input_image)
# apply canny transformation
canny_image = canny_transformation(lane_image)
# apply the mask
masked_image = region_of_interest(canny_image)
# detect lines using the Hough space
lines = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slop_intercept(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
# Combines the original image with the image with the detected lanes
final_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow("result", final_image)
cv2.waitKey(0)
"""


video = cv2.VideoCapture('.\image\\test2.mp4')

# crate a video file for the output video
frame_width=int(video.get(3))
frame_height=int(video.get(4))
size = (frame_width,frame_height)
output_video = cv2.VideoWriter('.\image\output.avi', cv2.VideoWriter_fourcc(*'MJPG'),10,size)

# Cycle trough the original video, frame by frame
while(video.isOpened()):
    ret, frame = video.read()
    # while the video is not ended
    if ret == True:
        # apply canny transformation
        canny_image = canny_transformation(frame)
        # apply the mask
        masked_image = region_of_interest(canny_image)
        # detect lines using the Hough space
        #lines = cv2.HoughLinesP(dst=masked_image, rho=2, theta=np.pi / 180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=5)
        lines = cv2.HoughLinesP(masked_image, lines=np.array([]),rho=2,theta=np.pi / 180, threshold=100,minLineLength=40,maxLineGap=5)
        averaged_lines = average_slop_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        # Combines the original image with the image with the detected lanes
        final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        output_video.write(final_image)
        cv2.imshow("result", final_image)
        if cv2.waitKey(1) == ord('q'):
            break
    # The video reached the last frame and ended
    else:
        print("video ended")
        break

video.release()
output_video.release()
cv2.destroyAllWindows()
