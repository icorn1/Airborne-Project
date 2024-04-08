import numpy as np
from shapely.geometry import Point, Polygon

# Define the parameters of the quarter circle
LOWER_LINE_HEIGHT = 380
UPPER_LINE_HEIGHT = 633
LINE_START = -724
LINE_END = 108
CIRCLE_START = -413
CIRCLE_END = -76
CIRCLE_RADIUS = 290


def create_working_region():
    center = (CIRCLE_START, UPPER_LINE_HEIGHT + CIRCLE_RADIUS)
    center2 = (CIRCLE_END, UPPER_LINE_HEIGHT + CIRCLE_RADIUS)
    theta = np.linspace(-np.pi / 2, 0, 10)  # Angle varies from pi/2 to 0 for quarter circle
    theta2 = np.linspace(-np.pi, -np.pi / 2, 10)  # Angle varies from pi/2 to 0 for quarter circle

    # Calculate the x and y coordinates of the points on the quarter circle
    x_circle = np.concatenate((center[0] + CIRCLE_RADIUS * np.cos(theta2),
                               center2[0] + CIRCLE_RADIUS * np.cos(theta)))
    y_circle = np.concatenate((center[1] + CIRCLE_RADIUS * np.sin(theta2),
                               center2[1] + CIRCLE_RADIUS * np.sin(theta)))

    # Define the vertices of the straight lines
    x_straight = np.array([LINE_START, LINE_END])
    y_straight = np.array([LOWER_LINE_HEIGHT, LOWER_LINE_HEIGHT])

    line1_x = np.array([LINE_START, x_circle[0]])
    line1_y = np.array([y_circle[0], y_circle[0]])

    line2_x = np.array([x_circle[-1], LINE_END])
    line2_y = np.array([y_circle[-1], y_circle[-1]])

    # Define the vertices of the polygon
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
