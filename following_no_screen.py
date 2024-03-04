import cv2
import numpy as np 

from read_matrices import read_intrinsics, read_R_T
from camera_class import Camera, rectify, undistort

from box_tracking import tracker
from live_classes import object_detector
from live_classes import gesture_recogniser

from feature_distance import match_distance

from comms import SerialCommunication
from control import ctl

def main(cameraL_id:int, cameraR_id:int, width: int, height: int, 
         port:str, baudrate:int = 9600, timeout:int = 1):
    
    object_model_path = "models/efficientdet_lite0.tflite"
    gesture_model_path = "models/gesture_recognizer.task"
    template_path = "images/cat_image_2.jpg"

    # get all the parameters for both cameras
    camera_matrix_L, distortion_L = read_intrinsics("camera_parameters/camera0_intrinsics.dat")
    R1, T1 = read_R_T("camera_parameters/camera0_rot_trans.dat")

    camera_matrix_R, distortion_R = read_intrinsics("camera_parameters/camera1_intrinsics.dat")
    R2, T2 = read_R_T("camera_parameters/camera1_rot_trans.dat")

    # finds the average focal_length for the depth calculation
    focal_lengths = [camera_matrix_L[0, 0], camera_matrix_L[1, 1], camera_matrix_R[0, 0], camera_matrix_R[1, 1]]
    avg_focal_length = np.average(focal_lengths)

    baseline = T2[0]

    # intialise both cameras
    left_cam = Camera(camera_matrix= camera_matrix_L, distortion_coefficients= distortion_L,
                      camera_id=cameraL_id, width=width, height=height)

    right_cam = Camera(camera_matrix= camera_matrix_R, distortion_coefficients= distortion_R,
                       camera_id=cameraR_id, width=width, height=height)
    
    mapxL, mapyL, mapxR, mapyR, image_ROI = rectify(height, width, left_cam.camera_matrix, left_cam.dist, 
                                                    right_cam.camera_matrix, right_cam.dist, R2, T2)
    _, _, new_width, new_height = image_ROI

    # object detection
    left_detect = object_detector(object_model_path)
    right_detect = object_detector(object_model_path)

    # box tracking
    left_tracker = tracker(template_path)
    right_tracker = tracker(template_path)

    # gesture recognition
    gr = gesture_recogniser(gesture_model_path)

    # initialise the distance matching and variables
    dc = match_distance()
    set_distance = 100    
    set_x_coord = int(new_width/2)

    distance = set_distance
    x_coord = set_x_coord
    
    target_center = int(new_width/2)
    matched_image = None

    # false means stop, true means go
    move_state = False

    line = SerialCommunication(port, baudrate, timeout)
    line.open_connection()
    controller = ctl(-2, 0.35, set_distance, set_x_coord)
    proximity_flag = False

    frame_interval = 10
    counter = 0

    left_objects = []
    right_objects = []

    target_found = False
    target_lost_counter = 0

    while left_cam.opened() and right_cam.opened:
        counter += 1

        measured_L_speed, measured_R_speed, proximity_flag = line.read_speeds()

        distorted_left = left_cam.read_frame()
        distorted_right = right_cam.read_frame()

        undistorted_left = undistort(distorted_left, mapxL, mapyL, image_ROI)
        undistorted_right = undistort(distorted_right, mapxR, mapyR, image_ROI)

        left_bboxes = left_detect.loop_function(undistorted_left)

        right_bboxes = right_detect.loop_function(undistorted_right)

        if counter % frame_interval == 0:
            counter = 0

            left_objects, left_corners = left_tracker.update(left_bboxes, undistorted_left)
            right_objects, right_corners = right_tracker.update(right_bboxes, undistorted_right)

        left_target_ID = left_tracker.get_target_ID()
        right_target_ID = right_tracker.get_target_ID()            
            
        if left_target_ID is not None and right_target_ID is not None:
            left_target_bbox = left_tracker.get_target_bbox()
            right_target_bbox = right_tracker.get_target_bbox()

            left_kp, left_des = left_tracker.get_object_kp_des()
            right_kp, right_des = right_tracker.get_object_kp_des()
            matched_image, coordinate_matches = dc.left_right_match(undistorted_left, undistorted_right, 
                                                                    left_target_bbox, right_target_bbox,
                                                                    left_kp, left_des, right_kp, right_des)
            
            if coordinate_matches is not None:
                distance, x_coord = dc.find_distances(coordinate_matches, baseline, avg_focal_length)
                
        if left_target_ID is not None:
            left_target_bbox = left_tracker.get_target_bbox()
            x, y, w, h = left_target_bbox
            target_center = int(x + w/2)
            
            target_crop = undistorted_left[y:y+h, x:x+w]
            recognition_frame, gestures = gr.loop_function(target_crop)

            if gestures:
                if gestures == "Thumb_Up":
                    move_state = True
                elif gestures == "Thumb_Down":
                    move_state = False

        if move_state:
            new_L_pwm, new_R_pwm = controller.compute_speeds(distance, x_coord)
        else:
            new_L_pwm, new_R_pwm = controller.compute_speeds(100, x_coord)

        print(f"Target Found: {target_found}")
        print(f"New Left PWM: {new_L_pwm},   New Right PWM: {new_R_pwm}")
        print(f"Distance: {distance}, Xcoord: {x_coord}, Move State: {move_state}")

        if not proximity_flag:
            if not target_found:
                target_lost_counter += 1
            else:
                target_lost_counter = 0
            
            if target_lost_counter < 30:
                line.write_speeds(int(new_L_pwm), int(new_R_pwm))
            else:
                line.write_speeds(0, 0)
        else:
            line.write_speeds(0, 0)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    left_cam.__del__()
    right_cam.__del__()

if __name__ == "__main__":
    main(2,0,640,480,"COM4",9600,1)