import cv2
import numpy as np


def generate_grid(contour):
    """
    Creates a 4x8 grid with a certain x and y spacing.
    The grid is centered around the found contour.
    Attaches an index to every point based on the specified numbering.

    :param contour: A single contour created by cv2.
    :return: Tuple containing the grid_points with attached indices.
    """
    cm = 9.625

    # Find the center of the contour
    M = cv2.moments(contour)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])

    # Generate a 4x8 grid with specified x and y spacing
    grid_rows, grid_cols = 4, 8
    x_spacing, y_spacing = 6 * cm, 40 / 6 * cm

    grid_points = []
    index = 1  # Starting index

    for j in range(grid_cols, 0, -1):
        for i in range(grid_rows, 0, -1):
            x = j * x_spacing
            y = i * y_spacing
            grid_points.append((x, y, index))
            index += 1

    # Convert the grid_points to NumPy array for further processing
    grid_points = np.array(grid_points)

    # Translate the grid points to the new center
    translation = np.array([int(centroid_x - np.mean(grid_points[:, 0])),
                            int(centroid_y - np.mean(grid_points[:, 1]))])

    grid_points[:, :2] += translation
    return grid_points


def optimize_grid(contour):
    """
    Optimizes the placement and rotation of a grid around a given contour.

    :param contour: A single contour created by cv2.
    :return: Tuple containing the maximum points inside the contour,
    the optimal translation vector, the optimal rotation angle and the grid points.
    """

    # Generate the grid
    grid_points = generate_grid(contour)

    # Initialize variables to store optimal translation and rotation
    optimal_translation_vector = np.zeros(2)
    optimal_rotation_angle = 0
    max_points_inside_contour = 0
    optimal_grid_points_inside_contour = []
    optimal_indices = []

    # Generate theta, dx and dy values (they start at 0)
    theta_values = np.concatenate((np.arange(0, 91, 5), np.arange(-91, -4, 5)))
    dx_values = np.concatenate((np.arange(0, 51, 5), np.arange(-51, -4, 5)))
    dy_values = np.concatenate((np.arange(0, 51, 5), np.arange(-51, -4, 5)))

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
            radius = 5
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
                if max_points_inside_contour > 31:
                    break
    return max_points_inside_contour, optimal_translation_vector, optimal_rotation_angle, grid_points, \
           optimal_grid_points_inside_contour, optimal_indices


def optimize_grid_angle(contour):
    """
    Optimizes the rotation of a grid around a given contour.

    :param contour: A single contour created by cv2.
    :return: Tuple containing the maximum points inside the contour, the optimal rotation angle and the grid points.
    """

    # Generate grid
    grid_points, grid_center_x, grid_center_y = generate_grid(contour)

    # Initialize variables to store the optimal rotation
    optimal_rotation_angle = 0
    max_points_inside_contour = 0

    # Generate theta values from 0 to 90 and from -85 to -90
    theta_values = np.concatenate((np.arange(0, 91, 5), np.arange(-91, -4, 5)))

    for theta in theta_values:
        rotation_matrix = cv2.getRotationMatrix2D((grid_center_x, grid_center_y), theta, 1)
        rotated_points = cv2.transform(np.array([grid_points]), rotation_matrix)[0]

        # Count the number of points inside the contour
        points_inside_contour = np.sum(np.array(
            [cv2.pointPolygonTest(contour, tuple(map(int, point)), False) >= -5 for point in
             rotated_points]))

        # Saves the angle if a new max points is reached. Also updates the current max.
        if points_inside_contour > max_points_inside_contour:
            max_points_inside_contour = points_inside_contour
            optimal_rotation_angle = theta

            # If the number of points inside the contour is higher than 10 the functions breaks (we're satisfied)
            if max_points_inside_contour > 31:
                break

    return max_points_inside_contour, optimal_rotation_angle, grid_points


def calculate_angles(angle):
    """
    Ensure the angle is within the range of 0 to 180 degrees.
    Handle the special case where the angle is 0 to avoid division by zero.
    Calculate and return two values, x and y, using a specific formula.

    :param angle: The input angle to be processed.
    :return: Tuple containing two values, x and y.
    """

    # Ensure angle is within the range of 0 to 180
    angle = (angle % 180 + 180) % 180

    # Avoid division by zero when angle is 0
    if angle == 0:
        return 0, np.pi

    ratio = 180 / angle
    x = np.sqrt(np.pi ** 2 / ratio)
    y = np.sqrt(np.pi ** 2 - x ** 2)
    return x, y


def get_ply_information(contour, T, show_plot=False):
    n, optimal_translation_vector, optimal_rotation_angle, grid_points, grid_points_inside_contour, optimal_indices = \
        optimize_grid(contour)
    translated_grid_points = grid_points[:, 0:2] + optimal_translation_vector
    grid_center_x = np.mean(grid_points[:, 0])
    grid_center_y = np.mean(grid_points[:, 1])
    pose_x = grid_center_x * T[0, 0] + T[0, 1]
    pose_y = -(grid_center_y * T[1, 0] + T[1, 1])  # Negate the y-coordinate
    robot_pose = [pose_x, pose_y]
    Rx, Ry = calculate_angles(optimal_rotation_angle)
    indices = np.zeros(32)
    indices[:len(optimal_indices)] = np.array(optimal_indices) + 1
    if show_plot:
        print(f"Number of ponts: {n}\n"
              f"Angle: {optimal_rotation_angle}, (x, y) : ({grid_center_x}, {grid_center_y})\n"
              f"Robot info: (x, y, z, Rx, Ry, Rz): {(robot_pose[0], robot_pose[1], 10, Rx, Ry, 0)}\n"
              f"Vacuum cups: {indices}")

        image_size = (600, 800)
        image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 0
        cv2.drawContours(image, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)
        rotation_matrix = cv2.getRotationMatrix2D(
            (grid_center_x + int(optimal_translation_vector[0]), grid_center_y + int(optimal_translation_vector[1])),
            optimal_rotation_angle, 1)
        final_rotated_grid_points = cv2.transform(np.array([translated_grid_points]), rotation_matrix)[0]
        for point in final_rotated_grid_points:
            cv2.circle(image, tuple(map(int, point)), 2, (255, 0, 0), -1)
        for point in grid_points_inside_contour:
            cv2.circle(image, tuple(map(int, point)), 5, (0, 255, 0), -1)

        cv2.imshow('Optimal Position, Rotation, and Grid Points Inside Contour', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    error_code = 0
    return robot_pose[0], robot_pose[1], Rx, Ry, error_code, indices
