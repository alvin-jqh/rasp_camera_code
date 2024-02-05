import numpy as np

def read_intrinsics(filename):
    """Reads the intrinsic matrix and distortion coefficients in the file, returns numpy arrays"""
    # open the reads the file, reading each line
    with open (filename, "r") as f:
        lines = f.readlines()

    # initialise each line
    intrinsic_matrix = np.zeros((3, 3)) 
    distortion_coeffs = np.zeros(5)
    matrix_section = None
    row_counter = 0

    # for each line in the data
    for line in lines:
        # removes any spaces
        line = line.strip()

        # switches to whichever matrix we need
        if line.startswith("intrinsic"):
            matrix_section = "intrinsic"
            row_counter = 0
        elif line.startswith("distortion"):
            matrix_section = "distortion"

        else:
            # converts the values into a list of floats
            values = list(map(float, line.split()))

            # fill in the matrix for intrinsic or distortion values
            if matrix_section == "intrinsic" and intrinsic_matrix is not None and len(values) == 3:
                intrinsic_matrix[row_counter, :] = values
                row_counter += 1
            elif matrix_section == "distortion" and len(values) == 5:
                distortion_coeffs[:] = values

    return intrinsic_matrix, distortion_coeffs

def read_R_T(filename):
    """Reads the rotation and translation matrix in the file, returns numpy arrays"""
    # open the reads the file, reading each line
    with open (filename, "r") as f:
        lines = f.readlines()

    # initialise each line
    rotation_matrix = np.zeros((3, 3)) 
    translation_matrix = np.zeros((3,1))
    matrix_section = None
    row_counter = 0

    # for each line in the data
    for line in lines:
        # removes any spaces
        line = line.strip()

        # switches to whichever matrix we need
        if line.startswith("R:"):
            matrix_section = "Rotation"
            row_counter = 0
        elif line.startswith("T:"):
            matrix_section = "Translation"
            row_counter = 0

        else:
            # converts the values into a list of floats
            values = list(map(float, line.split()))

            # fill in the matrix for the rotation or translation matrix
            if matrix_section == "Rotation" and rotation_matrix is not None and len(values) == 3:
                rotation_matrix[row_counter, :] = values
                row_counter += 1
            elif matrix_section == "Translation":
                translation_matrix[row_counter, 0] = values[0]
                row_counter += 1

    return rotation_matrix, translation_matrix

if __name__ == "__main__":
    camera_matrix_L, distortion_L = read_intrinsics("camera_parameters\camera0_intrinsics.dat")
    R1, T1 = read_R_T("camera_parameters\camera0_rot_trans.dat")

    camera_matrix_R, distortion_R = read_intrinsics("camera_parameters\camera1_intrinsics.dat")
    R2, T2 = read_R_T("camera_parameters\camera1_rot_trans.dat")

    print(f"Left camera matrix \n{camera_matrix_L}\nLeft Distortion coefficients \n{distortion_L}\n")
    print(f"Left Rotation\n{R1}\n, Left Translation\n{T1}\n")

    print(f"Right camera matrix \n{camera_matrix_R}\nRight Distortion coefficients \n{distortion_R}\n")
    print(f"Right Rotation\n{R2}\n, Right Translation\n{T2}")

    focal_lengths = [camera_matrix_L[0, 0], camera_matrix_L[1, 1], camera_matrix_R[0, 0], camera_matrix_R[1, 1]]
    print(np.average(focal_lengths))

    print(T2[0])