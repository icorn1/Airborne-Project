import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Variables
IMAGE_FOLDER = 'calibration_images_big'  # The folder where the calibration images are saved
ROBOT_POSES_FILENAME = 'robot_poses/grid_points.txt'  # The file where the robot poses are saved
INTRINSIC_MATRIX_FILENAME = 'calibration_matrices/IntrinsicMatrix.npz'  # The filename of the intrinsic matrix
DISTORTION_MATRIX_FILENAME = 'calibration_matrices/DistortionMatrix.npz'  # The filename of the distortion matrix
DETECTED_CORNERS_FOLDER = 'DetectedCorners'  # The filename where the detected corners images are saved to
TRANSLATION_MATRIX_FILENAME = 'calibration_matrices/Translation.npz'  # The filename of the tranlsation matrix
MASKING_THRESHOLD = 120  # The threshold for the mask
INVERT_MASK = 0  # 1 or 0 for inverting or not inverting the mask
GRID_SIZE = (8, 5)  # The grid shape that can be found by find_chessboard_corners
SQUARE_SIZE = 356.5 / 7  # The square size in mm
CHESSBOARD_DEBUG = False  # True if the images of the chessboard should be shown
CALCULATION_DEBUG = True  # True if the calulated x and y values should be shown with the MSE
CENTER_OFFSET = np.array(
    [-2, 298])  # The offset from the center of the calibration board to the center of the TOC in mm


def calculate_circle_locations(center):
    """
    Calculate the locations of inner corners of a checkerboard grid relative to a given center point.

    Parameters:
    - center (tuple): The coordinates (x, y) of the center point.

    Returns:
    - list: A list of tuples containing the coordinates of each inner corner.
    """
    # Calculate the half-width and half-height of the checkerboard
    half_width = (GRID_SIZE[0] + 1) * SQUARE_SIZE / 2
    half_height = (GRID_SIZE[1] + 1) * SQUARE_SIZE / 2

    # Initialize a list to store the coordinates of inner corners
    inner_corners = []

    # Iterate over the rows and columns of the grid to calculate the coordinates of each inner corner
    for i in range(GRID_SIZE[1], 0, -1):  # Start from the top row
        for j in range(1, GRID_SIZE[0] + 1):  # Move from left to right in each row
            # Calculate the x and y coordinates of the current inner corner relative to the center
            x = center[0] - half_width + j * SQUARE_SIZE
            y = center[1] - half_height + i * SQUARE_SIZE
            # Append the coordinates to the list of inner corners
            inner_corners.append((x, y))

    return inner_corners


def calculate_intrinsics(img_points, IndexWithImg, ImgSize):
    """
    Calculates the intrinsic camera parameters fx, fy, cx, cy from the images.

    Parameters:
    - img_points (list): List of detected corner points in the images.
    - IndexWithImg (list): List of indices corresponding to each image.
    - ImgSize (tuple): Size of the images (width, height).

    Returns:
    - tuple: Tuple containing the intrinsic matrix (mtx) and distortion coefficients (dist).
    """
    # Find the corners of the chessboard in the real world
    obj_points = []
    for i in range(len(IndexWithImg)):
        # Create a grid of points representing the corners of the chessboard in the real world
        objp = np.zeros((GRID_SIZE[0] * GRID_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:GRID_SIZE[0], 0:GRID_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
        obj_points.append(objp)

    # Find and save the intrinsic matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, ImgSize, None, None)
    # Save the intrinsic matrix and distortion coefficients to disk
    np.savez(INTRINSIC_MATRIX_FILENAME, mtx)
    np.savez(DISTORTION_MATRIX_FILENAME, dist)
    return mtx, dist


def undistort_image(img, dist, imx):
    """
    Undistorts an image using the provided camera matrix and distortion coefficients.

    Parameters:
    - img (numpy.ndarray): Input image to be undistorted.
    - dist (numpy.ndarray): Distortion coefficients.
    - imx (numpy.ndarray): Camera matrix.

    Returns:
    - numpy.ndarray: Undistorted image.
    """
    # Get the dimensions of the input image
    image_height, image_width = img.shape[:2]
    # Calculate the optimal new camera matrix and the region of interest (ROI)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(imx, dist, (image_width, image_height),
                                                      1, (image_width, image_height))
    # Undistort the input image using the camera matrix and distortion coefficients
    undistorted_img = cv2.undistort(img, imx, dist, None, newcameramtx)
    # Extract the ROI coordinates
    x, y, image_width, image_height = roi
    # Crop the undistorted image to the region of interest
    undistorted_img = undistorted_img[y:y + image_height, x:x + image_width]
    return undistorted_img


def find_chessboard_corners(images, dist, imx, distortion=True):
    """
    Finds the chessboard patterns and, if distortion is True, shows the images with the corners.

    Parameters:
    - images (list): List of input images.
    - dist (numpy.ndarray): Distortion coefficients.
    - imx (numpy.ndarray): Camera matrix.
    - distortion (bool): Flag indicating whether to undistort the images before processing.

    Returns:
    - tuple: Tuple containing the detected chessboard corners, indices of images with detected chessboards,
             and the undistorted version of the last processed image.
    """
    # Initializing some values
    chessboard_corners = []
    IndexWithImg = []
    undistorted = images[0]
    print("Finding corners...")

    # Iterate over each image
    for i, image in enumerate(images):
        undistorted = image
        # Undistort the image if required
        if distortion:
            undistorted = undistort_image(image, dist, imx)

        # Convert the image to grayscale
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Otsu's thresholding to obtain a binary mask
        _, otsu_mask = cv2.threshold(blur, 0, 255, INVERT_MASK + cv2.THRESH_OTSU)
        # Display intermediate images if CHESSBOARD_DEBUG is True
        if CHESSBOARD_DEBUG:
            resized_mask2 = cv2.resize(otsu_mask, (648, 512))
            resized_image = cv2.resize(image, (648, 512))
            resized_gray = cv2.resize(gray, (648, 512))
            cv2.imshow('mask adaptive', resized_mask2)
            cv2.imshow('img', resized_image)
            cv2.imshow('gray', resized_gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Configure parameters for SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.maxArea = 10e4
        detector = cv2.SimpleBlobDetector_create(params)
        # Find circles grid in the binary mask
        ret, corners = cv2.findCirclesGrid(otsu_mask, GRID_SIZE, cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector=detector)
        # If circles grid is found
        if ret:
            # Draw circles on the undistorted image and label them with their indices
            for idx, point in enumerate(corners):
                x_coord, y_coord = point[0][0], point[0][1]
                cv2.circle(undistorted, (int(x_coord), int(y_coord)), 5, (0, 0, 255), -1)
                cv2.putText(undistorted, str(idx), (int(x_coord) + 10, int(y_coord) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (0, 255, 0), 1)

            # Save an image with detected circles and their indices
            if not os.path.exists(DETECTED_CORNERS_FOLDER):
                os.mkdir(DETECTED_CORNERS_FOLDER)
            cv2.imwrite(f"{DETECTED_CORNERS_FOLDER}/{i:02d}.png", undistorted)
            # Display the undistorted image with detected circles if CHESSBOARD_DEBUG is True
            if CHESSBOARD_DEBUG:
                pic = cv2.resize(undistorted, (648, 512))
                cv2.imshow('Chessboard', pic)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Append the detected corners and index of the image to lists
            chessboard_corners.append(corners)
            IndexWithImg.append(i)
        else:
            print("No chessboard found in image: ", i)

        return chessboard_corners, IndexWithImg, undistorted


def save_translation_matrix(img_corners, poses):
    """
    Calculates and saves the translation matrix to convert image coordinates to robot frame coordinates.

    Parameters:
    - img_corners (list): List of detected circle locations in the images.
    - poses (list): List of robot poses (x, y, theta).

    Explanation:
    This function calculates the translation matrix to convert image coordinates to robot frame coordinates.
    It finds the circle locations in the robot frame, maps them to the image coordinates, and calculates
    the rotation angle with the lowest error. Then, it applies the rotation to align the coordinates.
    Finally, it applies the offset between the calibration board and TOC and saves the translation matrix.
    """
    # Convert robot poses to circle locations in the robot frame coordinates
    circle_locations = []
    for pose in poses:
        x, y = pose[1] * 1000, -pose[0] * 1000  # Convert from meters to millimeters and invert y-axis
        centre = (x, y)
        points = calculate_circle_locations(centre)
        circle_locations.append(points)

    # Extract x and y coordinates from the circle locations
    x_obj, y_obj = [], []
    for circle_location in circle_locations:
        x, y = zip(*circle_location)
        x_obj.append(x)
        y_obj.append(y)
    x_obj = np.array([item for sublist in x_obj for item in sublist])
    y_obj = np.array([item for sublist in y_obj for item in sublist])

    # Extract x and y coordinates from the detected image corners
    x_img, y_img = [], []
    for img_corner in img_corners:
        for point in img_corner:
            x_coord, y_coord = point[0][0], point[0][1]
            x_img.append(x_coord)
            y_img.append(y_coord)
    x_img = np.array(x_img)
    y_img = np.array(y_img)

    # Find the rotation angle with the lowest error
    angles = np.linspace(0, 2 * np.pi, num=20000)
    best_mse = 1.5
    best_angle = 0
    mse = []
    angies = []
    for angle in angles:
        # Rotate the image coordinates
        corrected_x_img = x_img * np.cos(angle) - y_img * np.sin(angle)
        corrected_y_img = x_img * np.sin(angle) + y_img * np.cos(angle)

        # Calculate the slope between the corrected image and robot values
        s_x = np.polyfit(corrected_x_img, x_obj, 1)
        s_y = np.polyfit(corrected_y_img, y_obj, 1)

        # Calculate the fit lines and mean square error
        fit_line_x = s_x[0] * corrected_x_img + s_x[1]
        fit_line_y = s_y[0] * corrected_y_img + s_y[1]
        mse_x = np.sqrt(np.mean((x_obj - fit_line_x) ** 2))
        mse_y = np.sqrt(np.mean((y_obj - fit_line_y) ** 2))
        calc_mse = (mse_x + mse_y) / 2

        # Update the best rotation angle and mean square error if a new best value is found
        if calc_mse < best_mse:
            mse.append(calc_mse)
            angies.append(angle)
            best_mse = calc_mse
            best_angle = angle

    # Apply the best found rotation angle
    corrected_x_img = x_img * np.cos(best_angle) - y_img * np.sin(best_angle)
    corrected_y_img = x_img * np.sin(best_angle) + y_img * np.cos(best_angle)

    # Calculate the slopes again after applying rotation
    s_x = np.polyfit(corrected_x_img, x_obj, 1)
    s_y = np.polyfit(corrected_y_img, y_obj, 1)

    # Calculate the fit lines and mean square error again
    fit_line_x = s_x[0] * corrected_x_img + s_x[1]
    fit_line_y = s_y[0] * corrected_y_img + s_y[1]
    mse_x = np.sqrt(np.mean((x_obj - fit_line_x) ** 2))
    mse_y = np.sqrt(np.mean((y_obj - fit_line_y) ** 2))

    # Apply the offset between the calibration board and TOC
    s_x[1] += CENTER_OFFSET[0]
    s_y[1] -= CENTER_OFFSET[1]

    # Create the translation matrix
    matrix = np.array([[s_x[0], s_x[1]], [s_y[0], s_y[1]]])

    # Save the translation matrix
    np.savez(TRANSLATION_MATRIX_FILENAME, matrix)

    if CALCULATION_DEBUG:
        print(f"MSE x-calibration: {mse_x}\n"
              f"MSE y-calibration: {mse_y}")

        plt.figure()
        plt.scatter(corrected_x_img, x_obj)
        plt.plot(corrected_x_img, fit_line_x, c='r')
        plt.xlabel("Image x-axis")
        plt.ylabel("Robot x-axis")

        plt.figure()
        plt.scatter(corrected_y_img, y_obj)
        plt.plot(corrected_y_img, fit_line_y, c='r')
        plt.xlabel("Image y-axis")
        plt.ylabel("Robot y-axis")
        plt.show()


def calibrate(image_folder, robot_pose_file, mtx_file, dst_file, distortion=True):
    """
    Calibrates the camera using images of a calibration pattern and robot poses.

    Parameters:
    - image_folder (str): Path to the folder containing calibration images.
    - robot_pose_file (str): Path to the file containing robot poses.
    - mtx_file (str): Path to the file containing the camera matrix.
    - dst_file (str): Path to the file containing distortion coefficients.
    - distortion (bool): Flag indicating whether to undistort the images before processing.

    Explanation:
    This function performs camera calibration using images of a calibration pattern and corresponding robot poses.
    It first loads the calibration images and robot poses. Then, it loads the camera matrix and distortion coefficients
    from files. Next, it finds chessboard corners in the images and calculates the translation matrix to convert
    image coordinates to robot frame coordinates. If distortion flag is False, it recalculates the camera matrix and
    distortion coefficients using the detected chessboard corners and recalculates the translation matrix accordingly.
    """
    # Load image files and robot poses
    image_files = sorted(glob.glob(f'{image_folder}/*.png'))
    images = [cv2.imread(f) for f in image_files]
    loaded_array = np.loadtxt(robot_pose_file)

    # Load camera matrix and distortion coefficients
    mtx_data = np.load(mtx_file)
    mtx = mtx_data['arr_0'].astype(np.float64)
    dst_data = np.load(dst_file)
    dst = dst_data['arr_0'].astype(np.float64)

    # Find chessboard corners in the images
    image_circles, IndexWithImg, img = find_chessboard_corners(images, dst, mtx, distortion=distortion)

    # If distortion is False, recalibrate the camera
    if not distortion:
        # Calculate new camera matrix and distortion coefficients
        mtx, dist = calculate_intrinsics(image_circles, IndexWithImg, images[0].shape[:2])
        # Find chessboard corners again with the new matrices
        image_circles, IndexWithImg, img = find_chessboard_corners(images, dst, mtx, distortion=True)

    # Extract robot poses corresponding to image with detected chessboards
    robot_poses = [loaded_array[i] for i in IndexWithImg]

    # Save the translation matrix
    save_translation_matrix(image_circles, robot_poses)


if __name__ == '__main__':
    calibrate(IMAGE_FOLDER, ROBOT_POSES_FILENAME, INTRINSIC_MATRIX_FILENAME, DISTORTION_MATRIX_FILENAME, 
              distortion=True)
