import cv2
import numpy as np

# Variables
VACUUM_CUP_RADIUS = 15  # mm
VACUUM_CUP_SPACING = 65  # mm
VACUUM_CUP_GRID = (4, 6)  # (y, x)
IMAGE_SIZE = (2015, 2549)  # Of the camera post undistortion
THETA_RANGE = (-91, 91)  # Optimization range
THETA_INTERVAL = 1  # Optimization interval
XY_RANGE = (-51, 51)  # Optimization range
XY_INTERVAL = 5  # Optimization interval
MAX_POINTS_INSIDE_CONTOUR = 23  # Breaks if higher than this number
SURROUNDING_RADIUS = -70


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

    for j in range(grid_cols, 0, -1):
        for i in range(0, grid_rows, 1):
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


def optimize_grid(contour, T):
    """
    Optimizes the placement and rotation of a grid around a given contour.

    :param T: The translation matrix
    :param contour: A single contour created by cv2.
    :return: Tuple containing the maximum points inside the contour,
    the optimal translation vector, the optimal rotation angle and the grid points.
    """

    # Generate the grid
    grid_points = generate_grid(contour, T)

    # Initialize variables to store optimal translation and rotation
    optimal_translation_vector = np.zeros(2)
    optimal_rotation_angle = 0
    max_points_inside_contour = 0
    optimal_grid_points_inside_contour = []
    optimal_indices = []

    # Generate theta, dx and dy values (they start at 0)
    theta_min, theta_max = THETA_RANGE
    xy_min, xy_max = XY_RANGE
    theta_values = np.concatenate((np.arange(0, theta_max, THETA_INTERVAL), np.arange(-theta_min, -1, THETA_INTERVAL)))
    dx_values = np.concatenate((np.arange(0, xy_max, XY_INTERVAL), np.arange(-xy_min, -1, XY_INTERVAL)))
    dy_values = np.concatenate((np.arange(0, xy_max, XY_INTERVAL), np.arange(-xy_min, -1, XY_INTERVAL)))

    # Generate all combinations of translation and rotation
    translation_vectors = np.array(np.meshgrid(dx_values, dy_values)).T.reshape(-1, 2)

    for theta in theta_values:
        for translation_vector in translation_vectors:
            translated_grid_points = grid_points[:, 0:2] + translation_vector
            grid_center_x = np.mean(grid_points[:, 0])
            grid_center_y = np.mean(grid_points[:, 1])

            rotation_matrix = cv2.getRotationMatrix2D(
                (grid_center_x + int(translation_vector[0]), grid_center_y + int(translation_vector[1])), theta, 1)
            rotated_points = cv2.transform(np.array([translated_grid_points]), rotation_matrix)[0]

            # Count the number of points inside the contour
            radius = VACUUM_CUP_RADIUS / T[0, 0]
            points_inside_contour = np.sum(np.array(
                [cv2.pointPolygonTest(contour, tuple(map(int, point)), True) >= radius for point in
                 rotated_points]))
            indices_and_points_inside_contour = [(index, point) for index, point in enumerate(rotated_points)
                                                 if cv2.pointPolygonTest(contour, tuple(map(int, point)),
                                                                         True) >= radius]
            indices, grid_points_inside_contour = zip(
                *indices_and_points_inside_contour) if indices_and_points_inside_contour else ([], [])

            # Saves the translation vector and angle if a new max points is reached. Also updates the current max.
            if points_inside_contour > max_points_inside_contour:
                max_points_inside_contour = points_inside_contour
                optimal_translation_vector = translation_vector
                optimal_rotation_angle = theta
                optimal_grid_points_inside_contour = grid_points_inside_contour
                optimal_indices = indices

                # If the number of points inside the contour is higher than 10 the functions breaks (we're satisfied)
                if max_points_inside_contour > MAX_POINTS_INSIDE_CONTOUR:
                    break
    return max_points_inside_contour, optimal_translation_vector, optimal_rotation_angle, grid_points, \
           optimal_grid_points_inside_contour, optimal_indices


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
    optimal_grid_points_inside_contour = []
    optimal_indices = []
    surrounding_indices = []
    surrounding_grid_points = []

    # Generate theta values from 0 to 90 and from -85 to -90
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
        indices_and_points_inside_contour = [(index, point) for index, point in enumerate(rotated_points)
                                             if cv2.pointPolygonTest(contour, tuple(map(int, point)),
                                                                     True) >= radius]
        indices, grid_points_inside_contour = zip(
            *indices_and_points_inside_contour) if indices_and_points_inside_contour else ([], [])

        indices_and_points_around_contour = [(index, point) for index, point in enumerate(rotated_points)
                                             if cv2.pointPolygonTest(contour, tuple(map(int, point)),
                                                                     True) >= SURROUNDING_RADIUS]
        indices_around, grid_points_around = zip(
            *indices_and_points_around_contour) if indices_and_points_around_contour else ([], [])

        # Saves the translation vector and angle if a new max points is reached. Also updates the current max.
        if points_inside_contour > max_points_inside_contour:
            max_points_inside_contour = points_inside_contour
            optimal_rotation_angle = theta * 180 / np.pi
            optimal_grid_points_inside_contour = grid_points_inside_contour
            optimal_indices = indices
            surrounding_indices = indices_around
            surrounding_grid_points = grid_points_around

            # If the number of points inside the contour is higher than 10 the functions breaks (we're satisfied)
            if max_points_inside_contour > MAX_POINTS_INSIDE_CONTOUR:
                break
    return max_points_inside_contour, optimal_rotation_angle, grid_points, surrounding_grid_points, \
           surrounding_indices


def rotate_grid_points(grid_points, contour, theta):
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
    n, optimal_rotation_angle, grid_points, grid_points_inside_contour, optimal_indices = \
        optimize_grid_angle(contour, T)

    # translated_grid_points = grid_points[:, 0:2] + optimal_translation_vector
    M = cv2.moments(contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])
    angle = -0.0328400808855735
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    grid_points = grid_points[:, :2].T
    grid_points = np.dot(rotation_matrix, grid_points).T

    grid_center_x = np.mean(grid_points[:, 0])
    grid_center_y = np.mean(grid_points[:, 1])

    print(f"Contour: {centroid_x, centroid_y}\n"
          f"Grid: {grid_center_x, grid_center_y}")

    pose_y = grid_center_x * T[0, 0] + T[0, 1]
    pose_x = -(grid_center_y * T[1, 0] + T[1, 1])  # Negate the y-coordinate
    robot_pose = np.array([pose_x, pose_y]) / 1000
    if show_plot:
        print(f"Number of ponts: {n}\n"
              f"Angle: {optimal_rotation_angle}, (x, y) : ({grid_center_x}, {grid_center_y})\n"
              f"Corrected robot info: (x, y, z, Rx, Ry, Rz): {(robot_pose[0], robot_pose[1], 10, 0, 0, optimal_rotation_angle / 180 * np.pi)}\n"
              f"Vacuum cups: {optimal_indices}")

        image = np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8) * 0
        cv2.drawContours(image, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)
        rotation_matrix = cv2.getRotationMatrix2D((grid_center_x, grid_center_y), optimal_rotation_angle, 1)
        final_rotated_grid_points = cv2.transform(np.array([grid_points]), rotation_matrix)[0]

        radius = abs(int(VACUUM_CUP_RADIUS / T[0, 0]))

        for point in final_rotated_grid_points:
            cv2.circle(image, tuple(map(int, point)), 2, (255, 0, 0), -1)
        for i in range(len(grid_points_inside_contour)):
            point = grid_points_inside_contour[i]
            index = optimal_indices[i]
            cv2.circle(image, tuple(map(int, point)), radius, (0, 255, 0), -1)
            cv2.putText(image, str(index), tuple(map(int, point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(image, (int(grid_center_x), int(grid_center_y)), 10, (255, 255, 255), -1)
        pic = cv2.resize(image, (1080, 853))
        cv2.imshow('Optimal Position, Rotation, and Grid Points Inside Contour', pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return robot_pose[0], robot_pose[1], optimal_rotation_angle, optimal_indices
