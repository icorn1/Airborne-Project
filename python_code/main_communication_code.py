from machine_vision_functions import get_ply_information
from calibrate import calibrate, undistort_image
from contours import find_most_similar_contour
from PLC_communication import write_values
from harvesters.core import Harvester
from time import sleep
import numpy as np
import socket
import snap7
import os
import cv2

# Variables
MASKING_THRESHOLD = 254
INVERT_MASK = 0
MINIMUM_CONTOUR_AREA = 200


def format_nums(integers):
    format_string = ", ".join(map(str, integers))
    return "({})".format(format_string)


def close_socket():
    client_socket.close()
    server_socket.close()


def get_genie_image():
    with Harvester() as h:
        h.add_file(CTI_FILE_PATH)
        h.update()
        with h.create(0) as ia:
            ia.start()
            with ia.fetch() as buffer:
                component = buffer.payload.components[0]
                width = component.width
                height = component.height
                bayer = component.data.reshape((height, width))
                img = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
    return img[..., ::-1]


def perform_calibration():
    robot_poses = np.loadtxt('robot_poses/custom_poses.txt')
    i = -1
    if i < len(robot_poses) - 1:
        i += 1
        client_socket.send(format_nums(([0])).encode())

        robot_pose = robot_poses[i]
        x, y, z, rx, ry, rz = robot_pose[:]
        client_socket.send(format_nums((x, y, z, rx, ry, rz)).encode())

        data = client_socket.recv(1024).decode()
        print("UR: ", data)

        sleep(2)
        # make picture and safe this picture
        image = get_genie_image()
        if not os.path.exists('calibration_images'):
            os.mkdir('calibration_images')
        cv2.imwrite(f'calibration_images/{i:02b}.png', image)

    else:
        client_socket.send(format_nums(([1])).encode())


def send_ply_information():
    ply_number = client_socket.recv(1024).decode()
    print("UR: ply_ID is ", ply_number, "\n")

    mtx_data = np.load('IntrinsicMatrix.npz')
    mtx = mtx_data['arr_0'].astype(np.float64)
    dst_data = np.load('DistortionMatrix.npz')
    dst = dst_data['arr_0'].astype(np.float64)
    T_data = np.load('Translation.npz')
    T = T_data['arr_0']

    image = get_genie_image()
    image = undistort_image(image, dst, mtx)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, MASKING_THRESHOLD, 255, INVERT_MASK)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > MINIMUM_CONTOUR_AREA]
    cv2.drawContours(image, contours, -1, (0, 255, 255), 3)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    min_index = find_most_similar_contour('contours', '1_mesh_contour.txt', contours, show_plot=False)
    x, y, rx, ry, error_code, cup_array = get_ply_information(contours[min_index], T, show_plot=False)

    print("PC: sending: x:", x, "| y:", y, "| Rz:", rx, "| Ry:", ry)

    client_socket.send(format_nums((x, y, rx, ry, error_code)).encode())
    print("PC: Data send to UR")

    if error_code == 0:
        plc = snap7.client.Client()
        plc.connect('192.168.0.1', 0, 1)  # IP address, rack, slot (from HW settings)
        print("PC: connecting to the PLC")
        print("PC: error_code: 0 \n")

        data = client_socket.recv(1024).decode()
        print("UR:", data, "vacuum cups \n")
        print("PC: activating vacuüm cups:", cup_array, "\n \n")

        write_values(plc, cup_array, 1)

        data = client_socket.recv(1024).decode()
        print("UR:", data, "vacuum cups \n")
        print("PC: Deactivate vacuüm cups:", cup_array, "\n \n")

        write_values(plc, cup_array, 0)

    if error_code == 1:
        print("PC: error_code: 1 \n \n")


"""
this is part of the initialisation when starting up the robot. This needs to be run only once.
"""
if __name__ == '__main__':
    host = '192.168.0.100'  # IP address of the UR controller (PC or laptop)
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

    while True:
        server_socket.listen(5)
        print("PC: Waiting for client connection...")
        client_socket, address = server_socket.accept()
        print("PC: Connected \n")
        task = client_socket.recv(1024).decode()

        if task == "calibration":
            print("UR: start calibration")
            perform_calibration()
            calibrate('calibration_images', 'robot_poses/custom_poses.txt', 'calibration_matrices/IntrinsicMatrix.npz',
                      'calibration_matrices/DistortionMatrix.npz', distortion=True)

        elif task == "moving":
            print("UR: start moving ply's")
            send_ply_information()
