import cv2
import numpy as np

def handle_target(mask):
    """Takes the mask of the target, returns the cetnroid"""
    target_centroid = None
    if mask is not None:
        # do some blurring to make the image less noisy
        mask = cv2.medianBlur(mask,5)
        mask = cv2.GaussianBlur(mask,(3,3),0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        if contours:
            # find the largest contour by area and save that only
            largest_contour = max(contours, key = cv2.contourArea)

            moments = cv2.moments(largest_contour)
            target_centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
    
    return target_centroid
                        