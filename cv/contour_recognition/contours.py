import cv2


def get_contours(img_inp, contour_inp):
    """Function for customizing contours visualization"""
    # Generate contours
    contours, hierarchy = cv2.findContours(img_inp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:

        # Area of each contour
        area = cv2.contourArea(cnt)

        # len(approx) == Number of angles of contour
        param = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * param, True)
        angles = len(approx)

        # Receive contours with proper area and angles
        if 9000 < area < 160000 and angles in (3, 4, 8):

            # Draw Contour
            # cv2.drawContours(contour_inp, cnt, -1, (255, 0, 255), 7)

            # Create Bounding Box coordinates for each contour
            x, y, w, h = cv2.boundingRect(approx)

            # Classify contours
            if angles == 3:
                cv2.putText(contour_inp, 'Triangle', (x+w//3, y+h//2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            elif angles == 4:
                cv2.putText(contour_inp, 'Rectangle', (x+w//3, y+h//2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(contour_inp, 'Circle', (x+w//3, y+h//2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

            # Possible variant to mark contours:
            # cv2.rectangle(contour_inp, (x, y), (x+w, y+h), (0, 255, 0), 5)
            # cv2.putText(contour_inp, 'Points: ' + str(len(approx)), (x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.7,
            #             (0, 255, 0), 2)
            # cv2.putText(contour_inp, 'Area: ' + str(int(area)), (x+w+20, y+45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
            #             (0, 255, 0), 2)
