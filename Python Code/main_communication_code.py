from machine_vision_functions import get_ply_information
from calibrate import calibrate, undistort_image
from contour_comparer import find_best_match
from PLC_communication import write_values
from laminate_data import LaminateStorage
from harvesters.core import Harvester
from contours import load_contour
from time import sleep
import warnings
import numpy as np
import socket
import snap7
import os
import cv2

# Variables
NUMBER_OF_VACUUM_CUPS = 24
DROPOFF_X = -.480
DROPOFF_Y = -.317
MASKING_THRESHOLD = 160
INVERT_MASK = 0
MINIMUM_CONTOUR_AREA = 10000
CTI_FILE_PATH = "C:/Program Files/Balluff/ImpactAcquire/bin/x64/mvGenTLProducer.cti"
SERIAL_NUMBER_CAM1 = 'S1129824'
SERIAL_NUMBER_CAM2 = 'S1126491'


def format_nums(integers):
    format_string = ", ".join(map(str, integers))
    return "({})".format(format_string)


def close_socket():
    client_socket.close()
    server_socket.close()


def get_genie_image(serial_number=SERIAL_NUMBER_CAM1):
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


def perform_calibration():
    robot_poses = np.loadtxt('grid_points.txt')

    for i, pose in enumerate(robot_poses):

        client_socket.send(format_nums(([0])).encode())

        robot_pose = robot_poses[i]
        x, y, z, rx, ry, rz = robot_pose[:]
        client_socket.send(format_nums((x, y, z, rx, ry, rz)).encode())

        data = client_socket.recv(1024).decode()
        print("UR: ", data)

        sleep(2)
        # make picture and safe this picture
        image = get_genie_image()
        if not os.path.exists('calibration_images_big'):
            os.mkdir('calibration_images_big')
        cv2.imwrite(f'calibration_images_big/{i:02d}.png', image)

    client_socket.send(format_nums(([1])).encode())
    calibrate('calibration_images', 'robot_pose2.txt', 'IntrinsicMatrix.npz',
              'DistortionMatrix.npz', (4, 4))


def send_ply_information(placed_plies=1):
    mtx_data = np.load('calibration_matrices/IntrinsicMatrix.npz')
    mtx = mtx_data['arr_0'].astype(np.float64)
    dst_data = np.load('calibration_matrices/DistortionMatrix.npz')
    dst = dst_data['arr_0'].astype(np.float64)
    T_data = np.load('calibration_matrices/Translation.npz')
    T = T_data['arr_0']
    for ply in laminate.ply_ids[placed_plies:]:
        # Error codes and their meanings:
        # 0 = there is no problem, the ply is found.
        # 1 = the laminate is finished
        # 2 = Wrong ply
        # 3 = No contours are found in the image
        error_code = 0

        image = get_genie_image()
        image = undistort_image(image, dst, mtx)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, MASKING_THRESHOLD, 255, INVERT_MASK)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > MINIMUM_CONTOUR_AREA]
        model_contour = load_contour(f"contours/{ply}_mesh_contour.txt")

        if len(contours) == 0:
            x, y, rotation, angle, cup_array_surrounding, cup_array = 0, 0, 0, 0, [], []
            error_code = 3
        else:
            index, angle, ret = find_best_match(contours, model_contour, image, show_plot=False)
            if not ret:
                x, y, rotation, angle, cup_array_surrounding, cup_array = 0, 0, 0, 0, [], []
                error_code = 2
            else:
                x, y, rotation, cup_array_surrounding, cup_array = get_ply_information(contours[index], T, show_plot=False)

        print(cup_array_surrounding)
        print(cup_array)
        cup_array_unused = [cup for cup in cup_array_surrounding if cup not in cup_array]

        # cv2.drawContours(image, contours, -1, (0, 255, 255), 3)
        # pic = cv2.resize(image, (1080, 858))
        #
        # cv2.imshow('img', pic)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        coordinates = laminate.coordinates[placed_plies]
        ply_angle = laminate.angles[placed_plies]
        robot_translation = np.array([coordinates[0] * T[0, 0],
                                      coordinates[1] * T[1, 0]]) / 1000

        xc = DROPOFF_X - robot_translation[1]
        yc = DROPOFF_Y - robot_translation[0]

        rz = rotation / 180 * np.pi
        rzc = (rotation + (ply_angle - (angle * 180 / np.pi))) / 180 * np.pi
        if rzc > np.pi:
            rzc -= np.pi * 2
        elif rzc < -np.pi:
            rzc += np.pi * 2

        print("PC: Sending error_code", error_code)
        client_socket.send(format_nums(([error_code])).encode())

        if error_code == 0:
            print("PC: sending: x:", x, "| y:", y, "| Rz:", rz,
                  "| xc:", xc, "| yc:", yc, "| Rzc:", rzc)
            client_socket.send(format_nums((x, y, rz, xc, yc, rzc)).encode())

            plc = snap7.client.Client()  # connecting to the PLC while the robot moves to ply
            plc.connect('192.168.0.1', 0, 1, 102)  # IP address, rack, slot (from HW settings)
            print("PC: connecting to the PLC")

            activation_code = client_socket.recv(1024).decode()
            print("UR:", activation_code, "vacuum cups \n")
            print("PC: activating unused vacuüm cups:", cup_array_unused, "\n")

            write_values(plc, cup_array_unused, 1)

            activation_code = client_socket.recv(1024).decode()
            print("UR:", activation_code, "vacuum cups \n")
            print("PC: activating used vacuüm cups:", cup_array, "\n")

            write_values(plc, cup_array, 1)

            all_vacuum_cups = list(range(0, NUMBER_OF_VACUUM_CUPS))
            activation_code = client_socket.recv(1024).decode()
            print("UR:", activation_code, "vacuum cups \n")
            print("PC: Deactivate vacuüm cups:", all_vacuum_cups, "\n")
            write_values(plc, all_vacuum_cups, 0)

            Error = client_socket.recv(1024).decode()
            if Error == '3':
                print("UR: pickup succesful")

                placed_plies += 1

            if Error == '4':
                print("UR: Failed pickup")
                activation_code = client_socket.recv(1024).decode()
                print("UR:", activation_code, "\n")
                send_ply_information(placed_plies)
                break

        elif error_code == 2:
            print("PC: Wrong ply")
            activation_code = client_socket.recv(1024).decode()
            print("UR:", activation_code, "\n")
            send_ply_information(placed_plies)

        elif error_code == 3:
            print("PC: No plies found")
            activation_code = client_socket.recv(1024).decode()
            print("UR:", activation_code, "\n")
            send_ply_information(placed_plies)

    # If the for loop ends, the laminate is finished.
    print("PC: Sending error_code 1 (Laminate finished)")
    error_code = 1
    client_socket.send(format_nums(([error_code])).encode())


"""
this is part of the initialisation when starting up the robot. This needs to be run only once.
"""
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    host = '192.168.0.11'  # IP address of the UR controller (PC or laptop)
    port = 50001  # The port used by the UR server
    server_socket = socket.socket(socket.AF_INET,
                                  socket.SOCK_STREAM)  # Create a socket object and bind the socket to the host and port
    server_socket.bind((host, port))

    """
    This while loop does the following:
        wait for a signal,
        accept signal,
        ... update this code ...
    """
    try:
        while True:
            server_socket.listen(5)
            print("PC: Waiting for client connection...")
            client_socket, address = server_socket.accept()
            print("PC: Connected \n")
            task = client_socket.recv(1024).decode()

            if task == "calibration":
                print("UR: start calibration")
                perform_calibration()

            elif task == "moving":
                print("UR: start moving ply's")
                Laminate_ID = client_socket.recv(1024).decode()
                laminate = LaminateStorage.load_from_pickle(f"laminates/{Laminate_ID}.pickle")

                # used when running send_ply_information2test()
                print("UR: Laminate ID is ", Laminate_ID, "\n")
                send_ply_information()
            else:
                print(task)
    except ConnectionResetError:
        print("Connection lost with UR")
