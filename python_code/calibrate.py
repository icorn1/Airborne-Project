import glob
import cv2
import numpy as np

# Variables
MASKING_THRESHOLD = 130  # The threshold for the mask
INVERT_MASK = 1  # 1 or 0 for inverting or not inverting the mask
FOUND_SQUARE_SIZE = (5, 5)  # The square size that can be found by find_chessboard_corners
ACTUAL_SQUARE_SIZE = (7, 7)  # The square size actual square size (Change the range values for different inner corners)
SQUARE_SIZE = 30  # mm


def calculate_inner_corner_coordinates(center):
    # Calculate the half-width and half-height of the checkerboard
    half_width = (ACTUAL_SQUARE_SIZE[0]) * SQUARE_SIZE / 2
    half_height = (ACTUAL_SQUARE_SIZE[1]) * SQUARE_SIZE / 2

    # Calculate the coordinates of each inner corner relative to the center
    inner_corners = []
    for i in range(1, ACTUAL_SQUARE_SIZE[0] - 1):
        for j in range(2, ACTUAL_SQUARE_SIZE[1]):
            x = center[0] - half_width + j * SQUARE_SIZE
            y = center[1] - half_height + i * SQUARE_SIZE
            inner_corners.append((x, y))
    return inner_corners


def undistort_image(img, dist, imx):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(imx, dist, (w, h), 1, (w, h))
    dist = cv2.undistort(img, imx, dist, None, newcameramtx)

    x, y, w, h = roi
    dist = dist[y:y + h, x:x + w]
    return dist


def find_chessboard_corners(images, dist, imx, distortion=True):
    """Finds the chessboard patterns and, if ShowImage is True, shows the images with the corners"""

    chessboard_corners = []
    IndexWithImg = []
    undistorted = images[0]
    i = 0
    print("Finding corners...")
    for image in images:
        undistorted = image
        if distortion:
            undistorted = undistort_image(image, dist, imx)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, MASKING_THRESHOLD, 255, INVERT_MASK)
        ret, corners = cv2.findChessboardCorners(mask, FOUND_SQUARE_SIZE, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                           cv2.CALIB_CB_FAST_CHECK +
                                                                           cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            chessboard_corners.append(corners)
            IndexWithImg.append(i)
            i = i + 1
        else:
            print("No chessboard found in image: ", i)
            i = i + 1
    return chessboard_corners, IndexWithImg, undistorted


def save_translation_matrix(img_corners, poses, print_mse=False):
    obj_corners = []
    for pose in poses:
        x, y = pose[0] * 1000, -pose[1] * 1000
        centre = (x, y)
        corners = calculate_inner_corner_coordinates(centre)
        obj_corners.append(corners)

    x_img, y_img = [], []
    for corner in img_corners:
        for point in corner:
            x_coord, y_coord = point[0][0], point[0][1]
            x_img.append(x_coord)
            y_img.append(y_coord)

    x_obj, y_obj = [], []
    for corner in obj_corners:
        x, y = zip(*corner)
        x_obj.append(x)
        y_obj.append(y)

    x_obj = np.array([item for sublist in x_obj for item in sublist])
    y_obj = np.array([item for sublist in y_obj for item in sublist])

    x_img = np.array(x_img)
    s_x = np.polyfit(x_img, x_obj, 1)
    fit_line_x = s_x[0] * x_img + s_x[1]
    mse_x = np.sqrt(np.mean((x_obj - fit_line_x) ** 2))

    y_img = np.array(y_img)
    s_y = np.polyfit(y_img, y_obj, 1)
    fit_line_y = s_y[0] * y_img + s_y[1]
    mse_y = np.sqrt(np.mean((y_obj - fit_line_y) ** 2))

    matrix = np.array([[s_x[0], s_x[1]], [s_y[0], s_y[1]]])
    np.savez('calibration_matrices/Translation.npz', matrix)
    if print_mse:
        print(f"MSE x-calibration: {mse_x}\n"
              f"MSE y-calibration: {mse_y}")


def calibrate(image_folder, robot_pose_file, mtx_file, dst_file, distortion=True):
    image_files = sorted(glob.glob(f'{image_folder}/*.png'))
    images = [cv2.imread(f) for f in image_files]
    loaded_array = np.loadtxt(robot_pose_file)
    mtx_data = np.load(mtx_file)
    mtx = mtx_data['arr_0'].astype(np.float64)
    dst_data = np.load(dst_file)
    dst = dst_data['arr_0'].astype(np.float64)

    image_corners, IndexWithImg, img = find_chessboard_corners(images, dst, mtx, distortion=distortion)
    if not distortion:
        mtx, dist = calculate_intrinsics(image_corners, IndexWithImg, FOUND_SQUARE_SIZE, 30, images[0].shape[:2])
    image_corners, IndexWithImg, img = find_chessboard_corners(images, dst, mtx)
    robot_poses = [loaded_array[i] for i in IndexWithImg]

    save_translation_matrix(image_corners, robot_poses)


def calculate_intrinsics(chessboard_corners, IndexWithImg, pattern_size, square_size, ImgSize):
    """Calculates the intrinc camera parameters fx, fy, cx, cy from the images"""

    # Find the corners of the chessboard in the image
    imgpoints = chessboard_corners
    # Find the corners of the chessboard in the real world
    objpoints = []
    for i in range(len(IndexWithImg)):
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        objpoints.append(objp)

    # Find the intrinsic matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ImgSize, None, None)
    np.savez("calibration_matrices/IntrinsicMatrixtest.npz", mtx)
    np.savez("calibration_matrices/DistortionMatrixtest.npz", dist)
    return mtx, dist


if __name__ == '__main__':
    calibrate('calibration_images', 'robot_poses/custom_poses.txt', 'calibration_matrices/IntrinsicMatrix.npz',
              'calibration_matrices/DistortionMatrix.npz', distortion=True)
