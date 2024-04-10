from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2

ACCURACY = 11  # Pixels
SCALING_FACTOR = 1.475  # The PNG generated from the DXF has a scale X times bigger than the images from the camera
ANGLE_OPTIMIZATION_ITERATIONS = 200  # The ammount of angles that are tried for finding the best angle. Affects the angle step size


def plot(img, cnts, fail_point_inds_per_cnt, trans_model_contour_points):
    """
    Plot contours, fail points, and model contours on the image.

    Parameters:
    - img (numpy.ndarray): Input image.
    - cnts (list): List of contours.
    - fail_point_inds_per_cnt (list): List of lists containing indices of fail points for each contour.
    - trans_model_contour_points (list): List of contours representing the transformed model.

    Explanation:
    This function plots contours, fail points, and model contours on the input image.
    Contours are drawn in blue, fail points in red, and model contours in blue.
    """
    plot_img = img.copy()
    model_color = (255, 0, 0)  # Blue (BGR format)
    fail_color = (0, 0, 255)  # Red

    # Draw contours and fail points
    for i, cnt in enumerate(cnts):
        # Assuming fail_point_inds_per_cnt contains indices of fail points for each contour
        fail_points = cnt[fail_point_inds_per_cnt[i]] if len(fail_point_inds_per_cnt[i]) > 0 else []
        for point in fail_points:
            cv2.circle(plot_img, tuple(point[0]), 5, fail_color, -1)  # Draw fail points

    # Convert each contour in trans_model_contour_points to the correct format
    for i in range(len(trans_model_contour_points)):
        trans_model_contour_points[i] = np.array(trans_model_contour_points[i], dtype=np.int32)

    # Draw contours on the image
    cv2.drawContours(plot_img, trans_model_contour_points, -1, model_color, 2)

    # Display the image
    pic = cv2.resize(plot_img, (1080, 858))
    cv2.imshow("Image", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_contour(filename):
    """
    Load a contour from a text file.

    :param filename: The name of the text file containing the contour data.
    :return: Loaded contour represented as a NumPy array with shape (-1, 1, 2).

    The text file should contain the contour data in a format compatible with NumPy's loadtxt function,
    such as comma-separated or space-separated values.
    """
    # Load the contour from the text file
    loaded_contour_reshaped = np.loadtxt(filename)

    # Reshape the loaded contour back to its original shape
    loaded_contour = loaded_contour_reshaped.reshape(-1, 1, 2)

    # Convert the loaded contour to integers
    loaded_contour = loaded_contour.astype(int)
    return loaded_contour


def get_orientation_pca(pts):
    """Function calculates the orientation of the object based on the angle of its first principal axes.

    Args:
        pts (n x 2 numpy array): xy points

    Returns:
        float, tuple: angle in radians, center point (x, y)
    """

    # Perform PCA analysis and calculate the center
    mean = np.empty(0)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts.reshape(-1, 2).astype(np.float64), mean)

    # Calculate the angle of the object
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    M = cv2.moments(pts.astype(np.int32))
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cntr = [cx, cy]

    return angle, cntr


def get_trans_matrix_and_transform(theta, center_point, center_shift, contour):
    """Generates transformation matrix for rotating contour points around
    a certain center point and shifting the center (translating the points)

    Args:
        theta (float): Angle of rotation in radians
        center_point (1x2 numpy array): Centerpoint of rotation: x, y
        center_shift (1x2 numpy array): Centerpoint shift: delta_x, delta_y
        contour (numpy array): Contour points as returned by OpenCV

    Returns:
        tuple: transformed contour, transformation matrix
    """

    trans_matrix = np.array([[np.cos(theta), -np.sin(theta), center_shift[0]],
                             [np.sin(theta), np.cos(theta), center_shift[1]]])
    cnt_trans = cv2.transform(contour - center_point, trans_matrix, -1) + center_point
    return cnt_trans, trans_matrix


def find_transform(model_contours, cnts, max_dev):
    """
    Find the transformation between model contours and target contours.

    Args:
        model_contours (list): List of model contours.
        cnts (list): List of target contours.
        max_dev (float): Maximum allowed deviation for matching points.

    Returns:
        tuple: Transformed target contours, optimal rotation angle, minimum sum of distances.

    Explanation:
    This function finds the transformation between the model contours and target contours.
    It calculates the orientation and center of the model and target contours, computes the angle
    difference and center shift between them, and applies a transformation to the model contours.
    It then iteratively optimizes the rotation angle to minimize the sum of distances between
    matched points. Finally, it returns the transformed target contours, optimal rotation angle,
    and minimum sum of distances.
    """
    # Calculate orientation and center of model and target contours
    theta_model, center_model = get_orientation_pca(model_contours[0])
    theta_part, center_part = get_orientation_pca(cnts[0])

    # Calculate angle difference and center shift
    theta = theta_part - theta_model
    center_shift = [center_part[0] - center_model[0], center_part[1] - center_model[1]]

    # Get initial transformed contours and transformation matrix
    cnt_trans, trans_matrix = get_trans_matrix_and_transform(theta, center_model, center_shift, model_contours[0])

    # Compute nearest neighbor distances
    distances, _ = NearestNeighbors(n_neighbors=1).fit(cnt_trans.reshape(-1, 2)).kneighbors(cnts[0].reshape(-1, 2))

    # Initialize minimum sum of distances and best angle
    min_sum = np.sum(distances)
    best_angle = theta

    # If max deviation is exceeded, perform angle optimization
    if max(distances) > max_dev:
        angles = np.linspace(theta - np.pi, theta + np.pi, num=ANGLE_OPTIMIZATION_ITERATIONS)

        # Iterate over angles for optimization
        for angle in angles:
            # Get transformed contours and transformation matrix for the current angle
            cnt_trans, trans_matrix = get_trans_matrix_and_transform(angle, center_model, center_shift, model_contours[0])

            # Compute nearest neighbor distances
            distances, _ = NearestNeighbors(n_neighbors=1).fit(cnt_trans.reshape(-1, 2)).kneighbors(cnts[0].reshape(-1, 2))

            # Update minimum sum of distances and best angle if improvement is found
            if np.sum(distances) < min_sum:
                min_sum = np.sum(distances)
                best_angle = angle

            # If max deviation condition is met, stop optimization
            if max(distances) < max_dev:
                break

    # Get final transformed contours and return results
    cnt_trans, trans_matrix = get_trans_matrix_and_transform(best_angle, center_model, center_shift, model_contours[0])
    return cnt_trans, best_angle, min_sum


def match_contour_to_model(cnts, model_contours, max_dev, img_ppmm, img, show_plot=False):
    """
    Check if the contours extracted from the image match the defined model contours.

    Args:
        cnts (list of numpy arrays): Contours obtained from the image.
        model_contours (list of numpy arrays): Defined model contours.
        max_dev (float): Maximum deviation from the model (in pixels).
        img_ppmm (float): Pixel per millimeter ratio for the image.
        img (numpy array): Image to overlay the results on.
        show_plot (bool): Whether to plot the results.

    Raises:
        ValueError: If the parameter figure_mode is not "show" or "return".

    Returns:
        tuple: Rotation angle, minimum sum of distances, pass/fail result, fail reason.
    """

    final_result = True
    fail_reason = " "

    # Check if the number of contours match
    if len(cnts) != len(model_contours):
        final_result = False
        fail_reason = "The number of contours do not match"

    # Scale model contours based on pixel per millimeter ratio
    model_contours_scaled = [model_contours[0] * img_ppmm]

    # Find the transformation between model and target contours
    cnt_trans, best_angle, min_sum = find_transform(model_contours_scaled, cnts, max_dev)

    # Initialize dictionary to store contour indices and corresponding model indices
    cnt_inds_and_mdl_inds = {0: 0}

    # Initialize list to store transformed model contours
    trans_model_contour_points = [cnt_trans.reshape(-1, 2)]

    # Initialize list to store indices of fail points for each contour
    fail_point_inds_per_cnt = [[]]

    # Check distances between contours and transformed model
    for cnt_ind in cnt_inds_and_mdl_inds:
        mdl_ind = cnt_inds_and_mdl_inds[cnt_ind]
        distances, _ = NearestNeighbors(n_neighbors=1).fit(
            trans_model_contour_points[mdl_ind]).kneighbors(cnts[cnt_ind].reshape(-1, 2))
        is_over = distances.reshape(1, -1)[0] > max_dev
        fail_point_inds_per_cnt[cnt_ind], = np.where(is_over)
        if len(fail_point_inds_per_cnt[cnt_ind]) > 0:
            if final_result:
                final_result = False
                fail_reason = "Exceeded maximum deviation"

    # Plot results if show_plot is True
    if show_plot:
        plot(img, cnts, fail_point_inds_per_cnt, [cnt_trans])

    return best_angle, min_sum, final_result, fail_reason


def find_best_match(contours, model_contour, img, show_plot=False):
    """
    Find the best-matching contour from a list of contours with respect to a given model contour.

    Args:
        contours (list of numpy arrays): List of contours to search through.
        model_contour (numpy array): Model contour to match against.
        img (numpy array): Image to overlay the results on.
        show_plot (bool): Whether to plot the results.

    Returns:
        tuple: Index of the best-matching contour, rotation angle, success flag.

    Explanation:
    This function iterates through a list of contours and matches each contour to the given model contour
    using the match_contour_to_model function. It returns the index of the best-matching contour, the rotation angle,
    and a success flag indicating whether a match was found.
    """
    best_score = float('inf')  # Initialize best score to positive infinity
    best_index = 0  # Initialize index of best-matching contour
    angle = 0  # Initialize rotation angle
    ret = False  # Initialize success flag

    # Return default values if contours list is empty
    if len(contours) == 0:
        return best_index, angle, ret

    # Iterate through each contour in the list
    for index, contour in enumerate(contours):
        # Match current contour to the model contour
        angle, score, ret, _ = match_contour_to_model([contour], [model_contour], ACCURACY, SCALING_FACTOR, img,
                                                      show_plot=show_plot)

        # If match is found, update best_index and exit loop
        if ret:
            best_index = index
            break

        # Update best_score and best_index if current score is lower
        if score < best_score:
            best_score = score
            best_index = index

    # Return index of best-matching contour, rotation angle, and success flag
    return best_index, angle, ret
