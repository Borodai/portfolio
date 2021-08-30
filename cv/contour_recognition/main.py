# Imports
import cv2
import numpy as np
from stacking import stack_images

# Constants
from constants import PARAMETERS, THRESHOLD_1, THRESHOLD_2, FILE_IN, FILE_OUT, CODEC
from constants import FRAMERATE, RESOLUTION, MASK, LOWER_R, LOWER_G, LOWER_B, HIGHER_R, HIGHER_G, HIGHER_B
from contours import get_contours


# Empty function to realise trackbar methods below
def empty():
    pass


def main():
    """Main function"""
    # Create track bar window to adjust thresholds for canny function
    cv2.namedWindow(PARAMETERS)
    cv2.resizeWindow(PARAMETERS, 640, 240)
    cv2.createTrackbar(THRESHOLD_1, PARAMETERS, 8, 255, empty)
    cv2.createTrackbar(THRESHOLD_2, PARAMETERS, 7, 255, empty)

    # Create track bar window to adjust mask levels
    cv2.namedWindow(MASK)
    cv2.resizeWindow(MASK, 640, 300)
    cv2.createTrackbar(LOWER_R, MASK, 0, 255, empty)
    cv2.createTrackbar(LOWER_G, MASK, 0, 255, empty)
    cv2.createTrackbar(LOWER_B, MASK, 186, 255, empty)
    cv2.createTrackbar(HIGHER_R, MASK, 255, 255, empty)
    cv2.createTrackbar(HIGHER_G, MASK, 255, 255, empty)
    cv2.createTrackbar(HIGHER_B, MASK, 255, 255, empty)

    # Open output video file
    cap = cv2.VideoCapture(FILE_IN)

    # Define output file object
    video_output = cv2.VideoWriter(FILE_OUT, CODEC, FRAMERATE, RESOLUTION)

    # Check if video file open
    if not cap.isOpened():
        print('Error opening video file')

    while cap.isOpened():
        success, img = cap.read()
        if success:

            # Create copy for proceed with result
            img_contour = img.copy()

            # Blur image to reduce noises
            img_blur = cv2.GaussianBlur(img, (9, 9), 3)

            # Convert image to HSV colors
            img_temp = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

            # Receive live settings for mask
            lower_R = cv2.getTrackbarPos(LOWER_R, MASK)
            lower_G = cv2.getTrackbarPos(LOWER_G, MASK)
            lower_B = cv2.getTrackbarPos(LOWER_B, MASK)
            higher_R = cv2.getTrackbarPos(HIGHER_R, MASK)
            higher_G = cv2.getTrackbarPos(HIGHER_G, MASK)
            higher_B = cv2.getTrackbarPos(HIGHER_B, MASK)

            # Apply mask
            lw = np.array([lower_R, lower_G, lower_B])
            hw = np.array([higher_R, higher_G, higher_B])
            mask = cv2.inRange(img_temp, lw, hw)

            # Convert to grey scale
            # img_grey = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

            # Receive values to adjust Canny function
            threshold1 = cv2.getTrackbarPos(THRESHOLD_1, PARAMETERS)
            threshold2 = cv2.getTrackbarPos(THRESHOLD_2, PARAMETERS)
            img_canny = cv2.Canny(mask, threshold1, threshold2)

            # Dilate result
            kernel = np.ones((6, 6))
            img_dilate = cv2.dilate(img_canny, kernel, iterations=1)

            # Call function to receive contours
            get_contours(img_dilate, img_contour)

            # Provide multiview streaming to operate with adjustments
            img_stack = stack_images(0.4, ([img, mask], [img_dilate, img_contour]))

            # Write output video
            video_output.write(img_contour)

            # Show video
            cv2.imshow('Video', img_stack)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
