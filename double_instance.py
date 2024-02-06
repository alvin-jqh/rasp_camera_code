import cv2
import numpy as np
import time

from read_matrices import read_intrinsics, read_R_T
from camera_class import Camera, rectify

from instance_tracking_class import instance_segmentor

from point_depth import calculate_depth


def main(cameraL_id:int, cameraR_id:int, width: int, height: int):

    object_model_path = "models\efficientdet_lite0.tflite"
    segmentor_model_path = "models\selfie_segmenter.tflite"

    # get all the parameters for both cameras
    camera_matrix_L, distortion_L = read_intrinsics("camera_parameters\camera0_intrinsics.dat")
    R1, T1 = read_R_T("camera_parameters\camera0_rot_trans.dat")

    camera_matrix_R, distortion_R = read_intrinsics("camera_parameters\camera1_intrinsics.dat")
    R2, T2 = read_R_T("camera_parameters\camera1_rot_trans.dat")

    # finds the average focal_length for the depth calculation
    focal_lengths = [camera_matrix_L[0, 0], camera_matrix_L[1, 1], camera_matrix_R[0, 0], camera_matrix_R[1, 1]]
    avg_focal_length = np.average(focal_lengths)

    baseline = -T2[0]

    # intialise both cameras
    left = Camera(camera_matrix= camera_matrix_L, distortion_coefficients= distortion_L,
                  camera_id=cameraL_id, width=width, height=height)

    right = Camera(camera_matrix= camera_matrix_R, distortion_coefficients= distortion_R,
                   camera_id=cameraR_id, width=width, height=height)
    
    left_inst = instance_segmentor(object_model_path,segmentor_model_path)
    right_inst = instance_segmentor(object_model_path,segmentor_model_path)

    # intialise the centroids
    L_target_centroid = None
    R_target_centroid = None

    while left.opened() and right.opened():
        left_image = left.read_frame()
        right_image = right.read_frame()

        # rectify and undistort the image based on the matrices produced from calibration
        undistorted_left, undistorted_right = rectify(left_image.copy(), right_image.copy(), 
                                                      left.camera_matrix, left.dist, 
                                                      right.camera_matrix, right.dist, R2, T2)

        left_detected_frame = left_inst.loop_function(undistorted_left)
        right_detected_frame = right_inst.loop_function(undistorted_right)

        L_target_centroid = left_inst.get_traget_centroid()
        R_target_centroid = right_inst.get_traget_centroid()

        if L_target_centroid is not None and R_target_centroid is not None:
            depth_cm = calculate_depth(L_target_centroid, R_target_centroid, avg_focal_length, baseline)
            cv2.putText(left_detected_frame, f"{depth_cm} cm", (L_target_centroid[0] + 20, L_target_centroid[1] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(right_detected_frame, f"{depth_cm} cm", (R_target_centroid[0] + 20, R_target_centroid[1] + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        if left_detected_frame is not None :
            cv2.imshow("Left", left_detected_frame)

        if right_detected_frame is not None:
            cv2.imshow("Right", right_detected_frame)

        # time.sleep(1)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    left.__del__()
    right.__del__()

if __name__ == "__main__":
    main(2,0,640,480)