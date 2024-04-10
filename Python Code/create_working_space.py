import numpy as np
from shapely.geometry import Point, Polygon

# Define the parameters of the quarter circle
LOWER_LINE_HEIGHT = 380  # The height of the lower line
UPPER_LINE_HEIGHT = 633  # The height of the upper line
LINE_START = -724  # The x start value of the working space
LINE_END = 108  # The x ending value of the working space
CIRCLE_START = -413  # The x value for the center of the left circle
CIRCLE_END = -76  # The x value for the center of the right circle
CIRCLE_RADIUS = 290  # The radius of both circles


def create_working_region():
    """
    Create a polygonal working region.

    Returns:
        Polygon: A Shapely Polygon object representing the working region.

    Explanation:
    This function defines a polygonal working region consisting of straight lines and quarter circle arcs.
    It calculates the coordinates of points on the quarter circle arcs and defines the vertices of the straight lines.
    These vertices are then used to create a polygon using the Shapely library.
    """
    center = (CIRCLE_START, UPPER_LINE_HEIGHT + CIRCLE_RADIUS)  # Define center point of the first quarter circle
    center2 = (CIRCLE_END, UPPER_LINE_HEIGHT + CIRCLE_RADIUS)   # Define center point of the second quarter circle

    # Define angles for quarter circle arcs
    theta = np.linspace(-np.pi / 2, 0, 10)   # Angle varies from -pi/2 to 0 for quarter circle
    theta2 = np.linspace(-np.pi, -np.pi / 2, 10)   # Angle varies from -pi to -pi/2 for quarter circle

    # Calculate x and y coordinates of points on the quarter circle arcs
    x_circle = np.concatenate((center[0] + CIRCLE_RADIUS * np.cos(theta2),
                               center2[0] + CIRCLE_RADIUS * np.cos(theta)))
    y_circle = np.concatenate((center[1] + CIRCLE_RADIUS * np.sin(theta2),
                               center2[1] + CIRCLE_RADIUS * np.sin(theta)))

    # Define vertices of the straight lines
    x_straight = np.array([LINE_START, LINE_END])
    y_straight = np.array([LOWER_LINE_HEIGHT, LOWER_LINE_HEIGHT])

    # Define vertices of the polygon
    line1_x = np.array([LINE_START, x_circle[0]])
    line1_y = np.array([y_circle[0], y_circle[0]])
    line2_x = np.array([x_circle[-1], LINE_END])
    line2_y = np.array([y_circle[-1], y_circle[-1]])

    polygon_vertices = np.vstack((np.concatenate((line1_x, x_circle, line2_x, x_straight[::-1])),
                                  np.concatenate((line1_y, y_circle, line2_y, y_straight[::-1])))).T

    # Create a Shapely Polygon object
    polygon = Polygon(polygon_vertices)

    return polygon


if __name__ == '__main__':
    poly = create_working_region()
    test_point = Point(0, 500)

    inside = test_point.within(poly)
    print(inside)
