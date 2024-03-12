import cv2
import numpy as np


def create_random_contour(angle=45, seed=0, show_plot=False):
    """
    Create a random squiggly contour and its rotated version within an image.

    :param angle: The angle of rotation for the generated contour (default is 45 degrees).
    :param seed: Seed for NumPy random number generation to reproduce results (default is 0).
    :param show_plot: Whether to display a plot showing the original and rotated contours (default is False).

    :return: Tuple containing the original and rotated contours.
    """
    # Create a white image
    image_size = (600, 800)
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 0

    # Generate a random squiggly shape
    np.random.seed(seed)
    num_points = 5
    squiggly_shape_points = np.random.randint(50, 450, size=(num_points, 2))

    # Convert the squiggly_shape_points to a closed contour
    squiggly_shape_contour = np.array([squiggly_shape_points], dtype=np.int32)
    cv2.fillPoly(image, squiggly_shape_contour, color=(255, 255, 255))

    # Find the contour of the squiggly shape (assuming a single contour)
    contours, _ = cv2.findContours(image[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    image_rot = rotate_image(image, angle)

    # Find the rotated contour of the squiggly shape (assuming a single contour)
    rots, _ = cv2.findContours(image_rot[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rot = rots[0]

    if show_plot:
        # Make a new images with just the contours
        image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 0
        cv2.drawContours(image, [rot], -1, (0, 0, 255), 3)
        cv2.drawContours(image, [contour], -1, (255, 0, 0), 3)

        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return contour, rot


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


def rotate_image(image, angle):
    """
    Rotate an input image by a specified angle.

    :param image: The input image to be rotated.
    :param angle: The angle of rotation in degrees.
    :return: Rotated image.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_contour(img, cnt, angle=0):
    """
    Rotate a contour within an image by a specified angle.

    :param img: The input image containing the contour.
    :param cnt: The contour to be rotated, represented as a list of points.
    :param angle: The angle of rotation in degrees (default is 0).
    :return: Image with the rotated contour filled.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # fill shape
    cv2.fillPoly(img, pts=cnt, color=(255, 255, 255))
    rect = cv2.minAreaRect(cnt[0])
    _, _, a = rect
    img = rotate_image(img, a + angle)
    return img


def normalize_img(img):
    """
    Normalize an input image by finding the external contour, cropping to the bounding rectangle,
    and resizing it to a fixed size (300x300).

    :param img: The input image to be normalized.
    :return: Normalized image with a fixed size of 300x300.
    """
    cnt, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rect = cv2.boundingRect(cnt[0])
    img_cropped_bounding_rect = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                                    bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]

    # resize all to same size
    img_resized = cv2.resize(img_cropped_bounding_rect, (300, 300))
    return img_resized


def compare_contours(cnt1, cnt2, show_plot=False):
    """
    Compare the similarity between two contours using the matchShapes method.

    :param cnt1: The first contour to be compared.
    :param cnt2: The second contour to be compared.
    :param show_plot: Whether to display a plot showing the original contours and their similarity score (default is False).

    :return: Similarity score between the two contours using matchShapes method.
    """
    image_size = (600, 800)

    img1 = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 0
    cv2.drawContours(img1, cnt1, -1, (0, 0, 255), 3)

    img2 = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 0
    cv2.drawContours(img2, cnt2, -1, (255, 0, 0), 3)

    rot_img1 = rotate_contour(img1, [cnt1])
    norm_img1 = normalize_img(rot_img1)
    cnt1, _ = cv2.findContours(norm_img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rot_img2 = rotate_contour(img2, [cnt2], angle=270)
    norm_img2 = normalize_img(rot_img2)
    cnt2, _ = cv2.findContours(norm_img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    similarity_score = cv2.matchShapes(cnt1[0], cnt2[0], cv2.CONTOURS_MATCH_I2, 0.0)

    if show_plot:
        print(f"The similarity score is : ""{:e}".format(similarity_score))
        image_size = (300, 300)
        image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 0
        cv2.drawContours(image, cnt1, -1, (0, 0, 255), 3)
        cv2.drawContours(image, cnt2, -1, (255, 0, 0), 3)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return similarity_score


def find_most_similar_contour(input_folder, contour_name, contours, show_plot=False):
    """
    Find the most similar contour to a loaded contour from a file in a given folder.

    :param input_folder: The folder containing the contour file.
    :param contour_name: The name of the contour file.
    :param contours: List of contours to compare against the loaded contour.
    :return: Index of the most similar contour in the provided list.
    :param show_plot: Whether to display a plot showing the original contours and their similarity score (default is False).
    """
    loaded_contour = load_contour(f'{input_folder}/{contour_name}')
    score = []
    for contour in contours:
        score.append(compare_contours(loaded_contour, contour, show_plot=show_plot))
    min_score = min(score)
    min_index = score.index(min_score)
    if show_plot:
        print(f"Lowest Score: {min_score} at index: {min_index}")
    return min_index


if __name__ == '__main__':
    contour1, contour2 = create_random_contour(angle=63, seed=1, show_plot=False)
    find_most_similar_contour('contours', '0_mesh_contour.txt', [contour1])
