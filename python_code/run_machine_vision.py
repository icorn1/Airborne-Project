import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from contours import find_most_similar_contour
from machine_vision_functions import get_ply_information
import timeit

# Create a black background image
image_size = (500, 500, 3)  # Width, Height, 3 channels (RGB)
img = np.zeros(image_size, dtype=np.uint8)

# Generate random points for polygons
points1 = np.array([[50, 50], [150, 100], [100, 200], [10, 32]], np.int32)
points2 = np.array([[250, 150], [350, 200], [300, 300], [143, 340]], np.int32)
points3 = np.array([[400, 50], [450, 150], [350, 100], [430, 120]], np.int32)

# Draw polygons on the black background
cv2.fillPoly(img, [points1], color=(255, 255, 255))
cv2.fillPoly(img, [points2], color=(255, 255, 255))
cv2.fillPoly(img, [points3], color=(255, 255, 255))

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)


# Display the image
# cv2.imshow('Circles Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def matrix_from_rtvec(rvec, tvec):
    M = np.eye(4)
    M[0:3, 0:3] = rvec
    M[0:3, 3] = tvec.squeeze()  # 1-D vector, row vector, column vector, whatever
    return M


def euler_to_matrix(axis_angle):
    """Convert Euler angles to rotation matrix."""
    return Rotation.from_rotvec(axis_angle).as_matrix()


def calculate_gripper2base(robot_pose):
    """Calculate R_gripper2base and t_gripper2base from robot pose."""
    position = np.array(robot_pose[:3])
    orientation = euler_to_matrix(robot_pose[3:])
    return orientation, position


file_data = np.load('FinalTransforms/T_gripper2cam_Method_0.npz')
M = file_data['arr_0']

min_index = find_most_similar_contour('contours', '0_mesh_contour.txt', contours)
get_ply_information(contours[min_index], M, show_plot=True)
# contour_time = timeit.timeit(lambda: find_most_similar_contour('contours', '0_mesh_contour.txt', contours), number=100)
# grid_time = timeit.timeit(lambda: get_ply_information(contours[min_index], M, show_plot=False), number=1)
#
# print(f"Contour time: {contour_time/100} s\n"
#       f"Grid time: {grid_time/1}")
