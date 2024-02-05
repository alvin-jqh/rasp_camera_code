import numpy as np

def find_depth(left_coord, right_coord, width, height, baseline, focal_length, FOV):
    """Takes coordinates in the left and right image that represent the same point in
       real space and calculates the depth of that point. """
    
    # converting focal length to pixels
    f_pixel = (width * 0.5)/ np.tan(FOV * 0.5 * np.pi/ 180)

    x_right = right_coord[0]
    x_left = left_coord[0]

    disparity = x_left - x_right

    zDepth = (baseline * f_pixel) / disparity

    return abs(zDepth)

def calculate_depth(left_centroid, right_centroid, focal_length, baseline):
    """Finds the disparity between two points and calculates the distance
    Args:
        left_centroid: centroid of the target in the left frame
        right_centroid: centorid of the target in the right frame
        focal_length: focal length of the cameras in pixels
        baseline: camera separation in cm
        
    Returns:
        zDepth: depth of that point in cm"""
    
    x_left = left_centroid[0]
    x_right = right_centroid[0]

    disparity = x_left - x_right

    zDepth = (baseline * focal_length) / disparity

    return abs(zDepth)
