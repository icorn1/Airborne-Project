# this python script opens a dxf file and tries to get the contours from it so that these contours can be matched with the used ply's

import matplotlib.pyplot as plt
import ezdxf
import cv2
import os
import numpy as np


def save_contour(filename, cont):
    """
    Save a contour to a text file.

    :param filename: The name of the text file to save the contour to.
    :param cont: The contour to be saved, represented as a NumPy array.
    """
    # Reshape the contour to a 2D array before saving
    contour_reshaped = cont.reshape(-1, 2)

    # Save the reshaped contour to a text file
    np.savetxt(filename, contour_reshaped)


def dfx2png(file, output_file):
    doc = ezdxf.readfile(file)
    msp = doc.modelspace()

    lines = [e for e in msp if e.dxftype() == "LINE"]
    lwpolylines = [e for e in msp if e.dxftype() == "LWPOLYLINE"]
    splines = [e for e in msp if e.dxftype() == "SPLINE"]
    circles = [e for e in msp if e.dxftype() == "CIRCLE"]

    if circles:
        for circle in circles:
            center = (circle.dxf.center.x, circle.dxf.center.y)
            radius = circle.dxf.radius
            plt.gca().add_patch(plt.Circle(center, radius, fill=False))

    if splines:
        for spline in splines:
            x_vals, y_vals, _ = zip(*spline.control_points)
            plt.plot(x_vals, y_vals)

    if lwpolylines:
        for lwpolyline in lwpolylines:
            x_vals, y_vals, _, _, _ = zip(*lwpolyline)
            x_vals += (x_vals[0],)
            y_vals += (y_vals[0],)
            plt.plot(x_vals, y_vals)

    if lines:
        for line in lines:
            start_point = (line.dxf.start.x, line.dxf.start.y)
            end_point = (line.dxf.end.x, line.dxf.end.y)
            plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]])

    plt.axis('equal')
    plt.axis('off')
    plt.savefig(f"{output_file}")


def png2contour(file, output_folder, debug=False):
    """
    Extract contours from PNG images and save them to text files.

    :param file: The file containing the ply contours
    :param output_folder: The folder where the generated contour text files will be saved.
    :param debug: If True, display images with drawn contours for debugging purposes (default is False).
    """
    # Get the image to the right state to find the contour
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 254, 255, 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # There should only be a single contour so this is fine
    for i, contour in enumerate(contours):
        if debug:
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save the contour
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        save_contour(f'{output_folder}/{i}_mesh_contour.txt', contour)


dfx2png('meshes/test_2Dfile.dxf', 'pngs/test2D.png')
png2contour('pngs/test2D.png', 'contours', debug=True)
