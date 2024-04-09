from contour_comparer import find_best_match, load_contour
from machine_vision_functions import get_ply_information
from create_working_space import create_working_region
from calibrate import calibrate, undistort_image
from PLC_communication import write_values
from laminate_data import LaminateStorage
from harvesters.core import Harvester
from shapely.geometry import Point
from time import sleep
import numpy as np
import warnings
import socket
import snap7
import os
import cv2

# Variables
IP_HOST = '192.168.0.11'
PORT_UR = 50001
IMAGE_FOLDER = 'calibration_images'  # The folder where the calibration images are saved
ROBOT_POSES_FILENAME = 'robot_poses/grid_points.txt'  # The file where the robot poses are saved
INTRINSIC_MATRIX_FILENAME = 'calibration_matrices/IntrinsicMatrix.npz'  # The filename of the intrinsic matrix
DISTORTION_MATRIX_FILENAME = 'calibration_matrices/DistortionMatrix.npz'  # The filename of the distortion matrix
DETECTED_CORNERS_FOLDER = 'DetectedCorners'  # The filename where the detected corners images are saved to
TRANSLATION_MATRIX_FILENAME = 'calibration_matrices/Translation.npz'  # The filename of the tranlsation matrix
CONTOUR_FOLDER = 'contours'
LAMINATES_FOLDER = 'laminates'
CTI_FILE_PATH = "C:/Program Files/Balluff/ImpactAcquire/bin/x64/mvGenTLProducer.cti"
SERIAL_NUMBER_CAM1 = 'S1129824'
SERIAL_NUMBER_CAM2 = 'S1126491'
NUMBER_OF_VACUUM_CUPS = 24
DROPOFF_X = -.480
DROPOFF_Y = -.317
Y_COLLISION_THRESHOLD = -.260
MASKING_THRESHOLD = 160
INVERT_MASK = 0
MINIMUM_CONTOUR_AREA = 10000
PLC_IP = '192.168.0.1'
DATABASE_RACK = 0
DATABASE_SLOT = 1
PLC_PORT = 102


def format_nums(values):
    """
    Format a list of integers into a string with comma-separated values enclosed in parentheses.

    Args:
        values: List of values to format.

    Returns:
        str: Formatted string with integers enclosed in parentheses and separated by commas.
    """
    format_string = ", ".join(map(str, values))
    return "({})".format(format_string)


def get_genie_image(serial_number=SERIAL_NUMBER_CAM1):
    """
    Get an image from a Genie camera.

    Args:
        serial_number (str): Serial number of the camera. Defaults to the value of SERIAL_NUMBER_CAM1.

    Returns:
        numpy.ndarray: Image from the Genie camera in BGR format.
    """
    with Harvester() as h:
        h.add_file(CTI_FILE_PATH)
        h.update()
        with h.create({'serial_number': serial_number}) as ia:
            ia.start()
            with ia.fetch() as buffer:
                component = buffer.payload.components[0]
                width = component.width
                height = component.height
                bayer = component.data.reshape((height, width))
                img = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
    return img[..., ::-1]


def get_image_and_contours(dst, mtx):
    """
    Capture image from the Genie camera, preprocess it, and find contours.

    Args:
        dst: Destination camera matrix for undistortion.
        mtx: Camera matrix for undistortion.

    Returns:
        Tuple containing the preprocessed image and its contours.
    """
    # Capture image from the Genie camera
    image = get_genie_image()
    # Undistort the image using the provided camera matrix
    image = undistort_image(image, dst, mtx)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary mask
    _, mask = cv2.threshold(gray, MASKING_THRESHOLD, 255, INVERT_MASK)
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Filter out contours with area smaller than the minimum contour area
    contours = [contour for contour in contours if cv2.contourArea(contour) > MINIMUM_CONTOUR_AREA]
    return image, contours


def activate_vacuum_cups(cup_array_unused, cup_array, client_socket):
    """
    Activate vacuum cups on the PLC and communicate with the client.

    Args:
        cup_array_unused: List of unused vacuum cups.
        cup_array: List of vacuum cups in use.
        client_socket: Socket object for communication with the client.
    """
    # Connecting to the PLC while the robot moves to ply
    plc = snap7.client.Client()
    plc.connect(PLC_IP, DATABASE_RACK, DATABASE_SLOT, PLC_PORT)  # IP address, rack, slot (from HW settings)
    print("PC: connecting to the PLC")

    # Receive activation code from the client socket
    activation_code = client_socket.recv(1024).decode()
    print("UR:", activation_code, "vacuum cups \n")
    print("PC: activating unused vacuüm cups:", cup_array_unused, "\n")

    # Write values to activate the unused vacuum cups
    write_values(plc, cup_array_unused, 1)

    # Receive activation code from the client socket
    activation_code = client_socket.recv(1024).decode()
    print("UR:", activation_code, "vacuum cups \n")
    print("PC: activating used vacuüm cups:", cup_array, "\n")

    # Write values to activate the used vacuum cups
    write_values(plc, cup_array, 1)

    # Generate a list of all vacuum cups
    all_vacuum_cups = list(range(0, NUMBER_OF_VACUUM_CUPS))

    # Receive activation code from the client socket
    activation_code = client_socket.recv(1024).decode()
    print("UR:", activation_code, "vacuum cups \n")
    print("PC: Deactivate vacuüm cups:", all_vacuum_cups, "\n")

    # Write values to deactivate all vacuum cups
    write_values(plc, all_vacuum_cups, 0)


def perform_calibration(client_socket):
    """
    Perform calibration using robot poses and images from a Genie camera.

    Args:
        client_socket: Socket object for communication with the client.

    Returns:
        None
    """
    # Load robot poses from file
    robot_poses = np.loadtxt(ROBOT_POSES_FILENAME)

    for i, pose in enumerate(robot_poses):
        # Send calibration command to the robot
        client_socket.send(format_nums(([0])).encode())

        # Send robot pose to the robot controller
        robot_pose = robot_poses[i]
        x, y, z, rx, ry, rz = robot_pose[:]
        client_socket.send(format_nums((x, y, z, rx, ry, rz)).encode())

        # Receive confirmation from the robot controller
        data = client_socket.recv(1024).decode()
        print("UR: ", data)

        # Capture image from the Genie camera and save it
        sleep(2)  # Delay to ensure stable image capture
        image = get_genie_image()
        if not os.path.exists(IMAGE_FOLDER):
            os.mkdir(IMAGE_FOLDER)
        cv2.imwrite(f'{IMAGE_FOLDER}/{i:02d}.png', image)

    # Send command to the robot controller to start calibration
    client_socket.send(format_nums(([1])).encode())
    # Perform calibration using the captured images and robot poses
    calibrate(IMAGE_FOLDER, ROBOT_POSES_FILENAME, INTRINSIC_MATRIX_FILENAME,
              DISTORTION_MATRIX_FILENAME, (4, 4))


def send_laminate_information(client_socket, laminate, placed_plies=0):
    """
    Send ply information to the client socket.

    Args:
        client_socket: Socket object for communication with the client.
        laminate: Laminate object containing ply information.
        placed_plies: Number of already placed plies.
    """
    # Load camera calibration data
    mtx_data = np.load('calibration_matrices/IntrinsicMatrix.npz')
    mtx = mtx_data['arr_0'].astype(np.float64)
    dst_data = np.load('calibration_matrices/DistortionMatrix.npz')
    dst = dst_data['arr_0'].astype(np.float64)
    T_data = np.load('calibration_matrices/Translation.npz')
    T = T_data['arr_0']

    # Loop through the remaining plies in the laminate
    for ply in laminate.ply_ids[placed_plies:]:
        # Initialize error code
        error_code = 0

        # Capture image and find contours
        image, contours = get_image_and_contours(dst, mtx)
        model_contour = load_contour(f"{CONTOUR_FOLDER}/{ply}_mesh_contour.txt")
        working_space = create_working_region()

        # Determine ply information based on contours
        if len(contours) == 0:
            x, y, rotation, angle, cup_array_surrounding, cup_array = 0, 0, 0, 0, [], []
            error_code = 3
        else:
            index, angle, ret = find_best_match(contours, model_contour, image, show_plot=False)
            if not ret:
                x, y, rotation, angle, cup_array_surrounding, cup_array = 0, 0, 0, 0, [], []
                error_code = 2
            else:
                x, y, rotation, cup_array_surrounding, cup_array = get_ply_information(contours[index], T,
                                                                                       show_plot=False)
                point = Point(y * 1000, x * 1000)
                print(point)
                inside = point.within(working_space)
                if not inside:
                    error_code = 4

        # Generate list of unused vacuum cups
        cup_array_unused = [cup for cup in cup_array_surrounding if cup not in cup_array]

        # Calculate robot translation and rotation
        coordinates = laminate.coordinates[placed_plies]
        ply_angle = laminate.angles[placed_plies]
        robot_translation = np.array([coordinates[0] * T[0, 0], coordinates[1] * T[1, 0]]) / 1000
        xc = DROPOFF_X - robot_translation[1]
        yc = DROPOFF_Y - robot_translation[0]
        rz = rotation / 180 * np.pi
        rzc = (rotation + (ply_angle - (angle * 180 / np.pi))) / 180 * np.pi
        if rzc > np.pi:
            rzc -= np.pi * 2
        elif rzc < -np.pi:
            rzc += np.pi * 2

        # Send error code to the client
        print("PC: Sending error_code", error_code)
        client_socket.send(format_nums(([error_code])).encode())

        # If no error, send ply information and activate vacuum cups
        if error_code == 0:
            collision_prevention_code = 0
            if y > Y_COLLISION_THRESHOLD:
                collision_prevention_code = 1
            print("PC: sending: x:", x, "| y:", y, "| Rz:", rz,
                  "| xc:", xc, "| yc:", yc, "| Rzc:", rzc, "| collision prevention code:", collision_prevention_code)
            client_socket.send(format_nums((x, y, rz, xc, yc, rzc, collision_prevention_code)).encode())
            activate_vacuum_cups(cup_array_unused, cup_array, client_socket)
            Error = client_socket.recv(1024).decode()
            if Error == '3':
                print("UR: pickup succesful")
                placed_plies += 1
            if Error == '4':
                print("UR: Failed pickup")
                activation_code = client_socket.recv(1024).decode()
                print("UR:", activation_code, "\n")
                send_laminate_information(client_socket, laminate, placed_plies)
                break
        else:
            # Handle error by recursively calling the function
            print(f"PC: Error code {error_code} detected.")
            activation_code = client_socket.recv(1024).decode()
            print("UR:", activation_code, "\n")
            send_laminate_information(client_socket, laminate, placed_plies)

    # If the for loop ends, the laminate is finished.
    print("PC: Sending error_code 1 (Laminate finished)")
    error_code = 1
    client_socket.send(format_nums(([error_code])).encode())


def main():
    """
    Main function to handle client connections and tasks.
    """
    host = IP_HOST  # IP address of the UR controller (PC or laptop)
    port = PORT_UR  # The port used by the UR server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a socket object
    server_socket.bind((host, port))  # Bind the socket to the host and port
    while True:
        server_socket.listen(5)
        print("PC: Waiting for client connection...")
        client_socket, address = server_socket.accept()
        print("PC: Connected \n")
        task = client_socket.recv(1024).decode()

        # Perform task based on client request
        if task == "calibration":
            print("UR: start calibration")
            perform_calibration(client_socket)
        elif task == "moving":
            print("UR: start moving ply's")
            Laminate_ID = client_socket.recv(1024).decode()
            laminate = LaminateStorage.load_from_pickle(f"{LAMINATES_FOLDER}/{Laminate_ID}.pickle")

            print("UR: Laminate ID is ", Laminate_ID, "\n")
            send_laminate_information(client_socket, laminate)
        else:
            print(task)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    try:
        main()
    except ConnectionResetError or ConnectionAbortedError:
        print("PC: Connection lost with UR")
