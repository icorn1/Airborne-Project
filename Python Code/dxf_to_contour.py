import matplotlib.pyplot as plt
import numpy as np
import ezdxf
import cv2
import os

# Variables
FIGURE_SIZE = (30, 30)  # The size of the created figure in matplotlib
MASKING_THRESHOLD = 254  # The masking threshold value
INVERT_MASKING = 1  # 1 or 0 for inverting or not inverting the mask
DXF_FILENAME = 'meshes/pastics.dxf'
PNG_FILENAME = 'pngs/palstics.png'
OUTPUT_CONTOUR_FOLDER = 'contours'


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
    """
    Convert a DXF file to a PNG image.

    Args:
        file (str): Path to the input DXF file.
        output_file (str): Path to save the output PNG file.

    Explanation:
    This function reads a DXF file using the ezdxf library and extracts various entities such as lines, polylines,
    splines, and circles. It then plots these entities using matplotlib and saves the resulting image as a PNG file.

    Note:
    - The function uses matplotlib to plot the DXF entities onto a plot.
    - The resulting PNG image will not have a transparent background.

    """
    doc = ezdxf.readfile(file)
    msp = doc.modelspace()

    # Extract entities from the DXF file
    lines = [e for e in msp if e.dxftype() == "LINE"]
    lwpolylines = [e for e in msp if e.dxftype() == "LWPOLYLINE"]
    splines = [e for e in msp if e.dxftype() == "SPLINE"]
    circles = [e for e in msp if e.dxftype() == "CIRCLE"]

    # Create a figure and axis for plotting
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_subplot(111)

    # Plot circles
    if circles:
        for circle in circles:
            center = (circle.dxf.center.x, circle.dxf.center.y)
            radius = circle.dxf.radius
            plt.gca().add_patch(plt.Circle(center, radius, fill=False))

    # Plot splines
    if splines:
        for spline in splines:
            x_vals, y_vals, _ = zip(*spline.control_points)
            ax.plot(x_vals, y_vals)

    # Plot lwpolylines
    if lwpolylines:
        for lwpolyline in lwpolylines:
            x_vals, y_vals, _, _, _ = zip(*lwpolyline)
            x_vals += (x_vals[0],)
            y_vals += (y_vals[0],)
            ax.plot(x_vals, y_vals)

    # Plot lines
    if lines:
        for line in lines:
            start_point = (line.dxf.start.x, line.dxf.start.y)
            end_point = (line.dxf.end.x, line.dxf.end.y)
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]])

    # Set plot aspect ratio and turn off axis
    ax.set_aspect('equal', 'box')
    plt.axis('off')

    # Save the plot as a PNG file
    plt.savefig(f"{output_file}", bbox_inches='tight')


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
    _, mask = cv2.threshold(gray, MASKING_THRESHOLD, 255, INVERT_MASKING)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    for i, contour in enumerate(contours):
        if debug:
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 3)
            pic = cv2.resize(img, (800, 600))
            cv2.imshow('img', pic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Save the contour
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        save_contour(f'{output_folder}/{i}_mesh_contour.txt', contour)


if __name__ == '__main__':
    dfx2png(DXF_FILENAME, PNG_FILENAME)
    png2contour(PNG_FILENAME, OUTPUT_CONTOUR_FOLDER, debug=True)
