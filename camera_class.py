import cv2
import numpy as np
import sys

class Camera:
    def __init__(self, camera_matrix = None, distortion_coefficients = None, camera_id:int = 0, width:int = 640, height:int = 480):
        """Creates a camera class, intialises and undistorts if given camera matrix and distortion coefficients"""
        self.camera_id = camera_id
        self.image_width = width
        self.image_height = height

        self.camera_matrix = camera_matrix
        self.dist = distortion_coefficients
        
        self.start()

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            sys.exit(f"Error: Unable to open webcam {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
    
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
    
    def read_frame(self):
        success, image = self.cap.read()
        if not success:
            sys.exit(
                "Error: Unable to read webcam"
            )
        
        image = cv2.flip(image, 1)

        return image
    
    def opened(self):
        if self.cap.isOpened():
            return True
        else:
            return False
    

def rectify(height, width, left_camera_matrix, left_dist, righ_camera_matrix, right_dist, R, T):
    """Returns the Maps and ROI for both the left and right camers after rectification"""

    R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(left_camera_matrix, left_dist, 
                                                       righ_camera_matrix, right_dist, 
                                                        (width, height), R, T,
                                                        flags = cv2.CALIB_FIX_INTRINSIC)
    
    mapxL, mapyL = cv2.initUndistortRectifyMap(left_camera_matrix, left_dist, R1, P1, (width, height), cv2.CV_16SC2)
    mapxR, mapyR = cv2.initUndistortRectifyMap(righ_camera_matrix, right_dist, R2, P2, (width, height), cv2.CV_16SC2)

    startL = (ROI1[0], ROI1[1])
    startR = (ROI2[0], ROI2[1])

    endL = (ROI1[0] + ROI1[2], ROI1[1] + ROI1[3])
    endR = (ROI2[0] + ROI2[2], ROI2[1] + ROI2[3])

    if startL[0] > startR[0]:
        x = startL[0]
    else: 
        x = startR[0]

    if startL[1] > startR[1]:
        y = startL[1]
    else: 
        y = startR[1]
    
    if endL[0] < endR[0]:
        x1 = endL[0]
    else: 
        x1 = endR[0]
    
    if endL[1] < endR[1]:
        y1 = endL[1]
    else: 
        y1 = endR[1]

    newROI = (x, y, x1-x, y1-y)

    return mapxL, mapyL, mapxR, mapyR, newROI

def undistort(image, mapx, mapy, ROI):
    """undistorts the image given the two maps and ROI"""
    undistorted_frame = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    x, y, w, h = ROI
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]
    # cv2.rectangle(undistorted_frame, (x,y), (x+w, y+h), (202, 125, 0), 2)

    return undistorted_frame


if __name__ == "__main__":
    Left = Camera(camera_id=0)
    Right = Camera(camera_id=1)

    while Left.cap.isOpened() and Right.cap.isOpened():
        Left_image = Left.read_frame()
        Right_image = Right.read_frame()

        cv2.imshow("Left", Left_image)
        cv2.imshow("Right", Right_image)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    Left.__del__()
    Right.__del__()