import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

ACCURACY = 15  # Pixels
SCALING_FACTOR = 1.46  # The PNG generated from the DXF has a scale X times bigger than the images from the camera
ANGLE_OPTIMIZATION_ITERATIONS = 200


def plot(img, cnts, fail_point_inds_per_cnt, trans_model_contour_points):
    model_color = "C1"
    object_color = "C0"
    fail_color = "r."
    plt.imshow(img, cmap="gray")
    for i, cnt in enumerate(cnts):
        plt.plot(cnt.reshape(-1, 2)[:, 0], cnt.reshape(-1, 2)[:, 1], color=object_color, label='Part')
        if len(fail_point_inds_per_cnt[0]) > 0:
            fail_points = cnt.reshape(-1, 2)[fail_point_inds_per_cnt[i], :]
            plt.plot(fail_points[:, 0], fail_points[:, 1], fail_color, markersize=5, label='Fail points')

    for to_object in [trans_model_contour_points]:
        to_object = np.append(to_object, [to_object[0]], axis=0)
        plt.plot(to_object.reshape(-1, 2)[:, 0], to_object.reshape(-1, 2)[:, 1], color=model_color, label='Model')

    plt.legend()
    plt.show()


def load_contour(filename):
    """
    Load a contour from a text file.

    :param filename: The name of the text file containing the contour data.
    :return: Loaded contour represented as a NumPy array with shape (-1, 1, 2).
    """
    # Load the contour from the text file
    loaded_contour_reshaped = np.loadtxt(filename)

    # Reshape the loaded contour back to its original shape
    loaded_contour = loaded_contour_reshaped.reshape(-1, 1, 2)

    # Convert the loaded contour to integers
    loaded_contour = loaded_contour.astype(int)
    return loaded_contour


def get_orientation_pca(pts):
    """Function calculates the orientation of the object based on the
    angle of its first principal axes.

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
    theta_model, center_model = get_orientation_pca(model_contours[0])
    theta_part, center_part = get_orientation_pca(cnts[0])
    theta = theta_part - theta_model
    center_shift = [center_part[0] - center_model[0], center_part[1] - center_model[1]]

    cnt_trans, trans_matrix = get_trans_matrix_and_transform(theta, center_model, center_shift,
                                                             model_contours[0])

    distances, _ = NearestNeighbors(n_neighbors=1).fit(
        cnt_trans.reshape(-1, 2)).kneighbors(cnts[0].reshape(-1, 2))

    min_sum = np.sum(distances)
    best_angle = theta
    if max(distances) > max_dev:
        angles = np.linspace(theta - np.pi, theta + np.pi, num=ANGLE_OPTIMIZATION_ITERATIONS)
        
        for angle in angles:
            cnt_trans, trans_matrix = get_trans_matrix_and_transform(angle, center_model, center_shift,
                                                                     model_contours[0])
            distances, _ = NearestNeighbors(n_neighbors=1).fit(
                cnt_trans.reshape(-1, 2)).kneighbors(cnts[0].reshape(-1, 2))

            if np.sum(distances) < min_sum:
                min_sum = np.sum(distances)
                best_angle = angle
                
            if max(distances) < max_dev:
                break
    cnt_trans, trans_matrix = get_trans_matrix_and_transform(best_angle, center_model, center_shift,
                                                             model_contours[0])
    return cnt_trans, best_angle, min_sum


def match_contour_to_model(cnts, model_contours, max_dev, img_ppmm, img, show_plot=False):
    """Checking if the contours got from the image match with the
    defined model contours.

    Args:
        cnts (list of numpy arrays): Contours list as returned by OpenCV
        model_contours (list of numpy arrays): Contours list as returned by OpenCV
        max_dev (float): Maximum deviation from the model (in pixels)
        img_ppmm (float): Pixel per millimeter ratio for the image.
        img (numpy array): Image to overlay the results on.
        show_plot (bool): Whether to plot the results or not.

    Raises:
        ValueError: If the parameter figure_mode is not "show" or "return"

    Returns:
        bool, numpy array: Did the part pass, the result image (None if figure_mode=="show")
    """

    final_result = True
    fail_reason = " "

    if len(cnts) != len(model_contours):
        final_result = False
        fail_reason = "The number of holes do not match"

    scale = img_ppmm 
    model_contours = model_contours
    model_contours[0] = model_contours[0] * scale

    cnt_trans, best_angle, min_sum = find_transform(model_contours, cnts, max_dev)

    cnt_trans = cnt_trans.reshape(-1, 2)
    cnt_inds_and_mdl_inds = {0: 0}
    trans_model_contour_points = [cnt_trans]

    # Going their through one by one and checking the distances.
    fail_point_inds_per_cnt = [[]]
    for cnt_ind in cnt_inds_and_mdl_inds:
        mdl_ind = cnt_inds_and_mdl_inds[cnt_ind]
        distances, _ = NearestNeighbors(n_neighbors=1).fit(
            trans_model_contour_points[mdl_ind]).kneighbors(cnts[cnt_ind].reshape(-1, 2))
        is_over = distances.reshape(1, -1)[0] > max_dev
        fail_point_inds_per_cnt[cnt_ind], = np.where(is_over)
        if len(fail_point_inds_per_cnt[0]) > 0:
            if final_result:
                final_result = False
                fail_reason = "Too big a deviation"

    if show_plot:
        plot(img, cnts, fail_point_inds_per_cnt, trans_model_contour_points)

    return best_angle, min_sum, final_result, fail_reason


def find_best_match(contours, model_contour, img, show_plot=False):
    best_score = float('inf')
    best_index = 0
    angle = 0
    ret = False

    if len(contours) == 0:
        return best_index, angle, ret

    for index, contour in enumerate(contours):
        angle, score, ret, reason = match_contour_to_model([contour], [model_contour], ACCURACY, SCALING_FACTOR, img,
                                                           show_plot=show_plot)
        if ret:
            angle = angle
            best_index = index
            break
        if score < best_score:
            best_score = score
            angle = angle
            best_index = index
    return best_index, angle, ret
