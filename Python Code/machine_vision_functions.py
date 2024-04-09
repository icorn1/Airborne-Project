import cv2
import numpy as np

# Variables
VACUUM_CUP_RADIUS = 15  # mm
VACUUM_CUP_SPACING = 65  # mm
VACUUM_CUP_GRID = (4, 6)  # (y, x)
IMAGE_SIZE = (2015, 2549)  # Of the camera post undistortion
THETA_RANGE = (-91, 91)  # Optimization range
THETA_INTERVAL = 1  # Optimization interval
MAX_POINTS_INSIDE_CONTOUR = 23  # Breaks if higher than this number
SURROUNDING_RADIUS = -70  # the distance where a vacuum cup is considered surrounding in pixels
ANGLE = -0.0328400808855735  # The angle between the robot and the camera in rad


def generate_grid(contour, T):
    """
    Creates a 4x8 grid with a certain x and y spacing.
    The grid is centered around the found contour.
    Attaches an index to every point based on the specified numbering.

    :param T: The translation matrix
    :param contour: A single contour created by cv2.
    :return: Tuple containing the grid_points with attached indices.
    """

    # Find the center of the contour
    M = cv2.moments(contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    # Generate a 4x8 grid with specified x and y spacing
    grid_rows, grid_cols = VACUUM_CUP_GRID
    x_spacing, y_spacing = VACUUM_CUP_SPACING / T[0, 0], VACUUM_CUP_SPACING / T[1, 0]
    grid_points = []
    index = 1  # Starting index

    for j in range(0, grid_cols, 1):
        for i in range(grid_rows, 0, -1):
            x = i * x_spacing
            y = j * y_spacing
            grid_points.append((x, y, index))
            index += 1

    # Convert the grid_points to NumPy array for further processing
    grid_points = np.array(grid_points)

    # Translate the grid points to the new center
    translation = np.array([int(centroid_x - np.mean(grid_points[:, 0])),
                            int(centroid_y - np.mean(grid_points[:, 1]))])

    grid_points[:, :2] += translation
    return grid_points


def optimize_grid_angle(contour, T):
    """
    Optimizes the rotation of a grid around a given contour.

    :param T: The translation matrix
    :param contour: A single contour created by cv2.
    :return: Tuple containing the maximum points inside the contour, the optimal rotation angle and the grid points.
    """

    # Generate grid
    grid_points = generate_grid(contour, T)

    # Initialize variables to store the optimal rotation
    optimal_rotation_angle = 0
    max_points_inside_contour = 0
    inside_grid_points = []
    inside_indices = []
    surrounding_indices = []
    surrounding_grid_points = []

    # Generate theta values
    theta_min, theta_max = THETA_RANGE
    theta_values = np.concatenate((np.arange(0, theta_max / 180 * np.pi, THETA_INTERVAL / 180 * np.pi),
                                   np.arange(-theta_min / 180 * np.pi, -1, THETA_INTERVAL / 180 * np.pi)))

    for theta in theta_values:
        rotated_points = rotate_grid_points(grid_points, contour, theta)

        # Count the number of points inside the contour
        radius = VACUUM_CUP_RADIUS / T[0, 0]
        points_inside_contour = np.sum(np.array(
            [cv2.pointPolygonTest(contour, tuple(map(int, point)), True) >= radius for point in
             rotated_points]))

        # Get the indices of the points inside the contour
        indices_and_points_inside_contour = [(index, point) for index, point in enumerate(rotated_points)
                                             if cv2.pointPolygonTest(contour, tuple(map(int, point)),
                                                                     True) >= radius]
        indices_inside, grid_points_inside = zip(
            *indices_and_points_inside_contour) if indices_and_points_inside_contour else ([], [])

        # Get the indices and points surrounding the contour
        indices_and_points_around_contour = [(index, point) for index, point in enumerate(rotated_points)
                                             if cv2.pointPolygonTest(contour, tuple(map(int, point)),
                                                                     True) >= SURROUNDING_RADIUS]
        indices_around, grid_points_around = zip(
            *indices_and_points_around_contour) if indices_and_points_around_contour else ([], [])

        # Saves the and angle if a new max points is reached. Also updates the current max.
        if points_inside_contour > max_points_inside_contour:
            max_points_inside_contour = points_inside_contour
            optimal_rotation_angle = theta * 180 / np.pi
            inside_indices = indices_inside
            inside_grid_points = grid_points_inside
            surrounding_indices = indices_around
            surrounding_grid_points = grid_points_around

            # If the number of points inside the contour is higher than the threshold the functions breaks
            if max_points_inside_contour > MAX_POINTS_INSIDE_CONTOUR:
                break
    return max_points_inside_contour, optimal_rotation_angle, grid_points, surrounding_grid_points, \
           surrounding_indices, inside_grid_points, inside_indices


def rotate_grid_points(grid_points, contour, theta):
    """
    Rotate a set of grid points around the centroid of a contour by a given angle.

    Args:
        grid_points (numpy array): Array of grid points with shape (n, 2).
        contour (numpy array): Contour representing the object with shape (n, 1, 2).
        theta (float): Angle of rotation in radians.

    Returns:
        numpy array: Rotated grid points.

    """
    # Translation to the centroid of the grid points
    translation = np.array([int(np.mean(grid_points[:, 0])),
                            int(np.mean(grid_points[:, 1]))])
    zerod_grid_points = grid_points[:, :2] - translation

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Rotate the points
    points = zerod_grid_points[:, :2].T
    rotated_points = np.dot(rotation_matrix, points).T

    # Calculate centroid of the contour
    M = cv2.moments(contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    # Translation to the centroid of the rotated points
    translation = np.array([int(centroid_x - np.mean(rotated_points[:, 0])),
                            int(centroid_y - np.mean(rotated_points[:, 1]))])

    # Apply translation
    rotated_points[:, :2] += translation

    return rotated_points


def get_ply_information(contour, T, show_plot=False):
    """
    Get information about the PLY (Polygonal) object based on its contour and transformation matrix.

    Args:
        contour (numpy array): Contour representing the object.
        T (numpy array): Transformation matrix.
        show_plot (bool): Whether to display the plot or not.

    Returns:
        tuple: Tuple containing the robot pose (x, y, angle), surrounding vacuum cup indices, and inside vacuum cup indices.
    """
    # Optimize grid angle
    n, optimal_rotation_angle, grid_points, surrounding_grid_points, surrounding_indices, inside_grid_points, inside_indices = \
        optimize_grid_angle(contour, T)

    # Rotate grid points based on optimal rotation angle
    rotation_matrix = np.array([[np.cos(ANGLE), -np.sin(ANGLE)],
                                [np.sin(ANGLE), np.cos(ANGLE)]])
    grid_points = grid_points[:, :2].T
    grid_points = np.dot(rotation_matrix, grid_points).T

    # Calculate centroid of rotated grid points
    grid_center_x = np.mean(grid_points[:, 0])
    grid_center_y = np.mean(grid_points[:, 1])

    # Calculate robot pose
    pose_y = grid_center_x * T[0, 0] + T[0, 1]
    pose_x = -(grid_center_y * T[1, 0] + T[1, 1])  # Negate the y-coordinate
    robot_pose = np.array([pose_x, pose_y]) / 1000  # Convert to meters

    if show_plot:
        # Display plot with information
        image = np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8) * 0
        cv2.drawContours(image, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)
        final_rotated_grid_points = rotate_grid_points(grid_points, contour, optimal_rotation_angle / 180 * np.pi)
        radius = abs(int(VACUUM_CUP_RADIUS / T[0, 0]))

        # Plot grid points
        for point in final_rotated_grid_points:
            cv2.circle(image, tuple(map(int, point)), 2, (255, 0, 0), -1)

        # Plot surrounding vacuum cup indices
        for i in range(len(surrounding_grid_points)):
            point = surrounding_grid_points[i]
            index = surrounding_indices[i]
            cv2.circle(image, tuple(map(int, point)), radius, (0, 255, 0), -1)
            cv2.putText(image, str(index), tuple(map(int, point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Plot inside vacuum cup indices
        for i in range(len(inside_grid_points)):
            point = inside_grid_points[i]
            index = inside_indices[i]
            cv2.circle(image, tuple(map(int, point)), radius, (0, 255, 255), -1)
            cv2.putText(image, str(index), tuple(map(int, point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.circle(image, (int(grid_center_x), int(grid_center_y)), 10, (255, 255, 255), -1)
        pic = cv2.resize(image, (1080, 853))
        cv2.imshow('Optimal Position, Rotation, and Grid Points Inside Contour', pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return robot_pose[0], robot_pose[1], optimal_rotation_angle, surrounding_indices, inside_indices
