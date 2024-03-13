import socket
import numpy as np
import os
import telnetlib
from ftplib import FTP
import cv2
from time import sleep


def format_nums1(integers):
    format_string = "({}, {}, {}, {}, {})".format(integers[0], integers[1],
                                                  integers[2], integers[3],
                                                  integers[4])
    return format_string


def format_nums2(integers):
    format_string = "({}, {}, {}, {}, {}, {})".format(integers[0], integers[1], 
                                                      integers[2], integers[3], 
                                                      integers[4], integers[5])
    return format_string


def format_nums3(integers):
    format_string = "({})".format(integers[0])
    return format_string


def close_socket():
    client_socket.close()
    server_socket.close()


def camera_data():
            #the camera gathers information. in this case some randon coordinates
    x =-0.4
    y =-0.5
    rx = 3.14
    ry = 0
    error_code =0  # Error_code 1 tells if the ply with the ply_ID given by the UR could be found or if a ply diviates more than 80% or something.
                    # Error_code 0 tells that there is no problem.
    cup_array = [0] * 24
    return x, y, rx, ry, error_code, cup_array


def save_cognex_image(output_folder, file_index):
    # cognex's config
    ip = "192.168.0.10"
    user = 'admin'
    password = ''

    # telnet login
    tn = telnetlib.Telnet(ip, 10000)
    telnet_user = user + '\r\n'
    tn.write(telnet_user.encode('ascii'))
    tn.write("\r\n".encode('ascii'))

    # capture
    tn.write(b"SE8\r\n")

    # ftp login
    ftp = FTP(ip)
    ftp.login(user)

    # show all file in cognex
    ftp.dir()

    # download file from cognex
    filename = 'image.bmp'
    lf = open(filename, "wb")
    ftp.retrbinary("RETR " + filename, lf.write)
    lf.close()

    image = cv2.imread('image.bmp')


    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    cv2.imwrite(f'{output_folder}/{file_index}.png', image)


"""
this is part of the initialisation when starting up the robot. This needs to be run only once.
"""   
host ='192.168.0.100'       # IP address of the UR controller (PC or laptop)
port =50003                 # The port used by the UR server
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # Create a socket object and bind the socket to the host and port
server_socket.bind((host,port))

robot_poses = np.loadtxt('calib/robot_pose2.txt')
i = -1


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


        if i < len(robot_poses) - 1:
            i += 1
            client_socket.send(format_nums3(([0])).encode()) 
            
            robot_pose = robot_poses[i]
            x, y, z, rx, ry, rz = robot_pose[:]
            client_socket.send(format_nums2((x, y, z, rx, ry, rz)).encode())

            data = client_socket.recv(1024).decode()
            print("UR: ", data)

            sleep(2)
            # make picture and safe this picture
            save_cognex_image('calibration_images', i)


        else:
            i = -1
            client_socket.send(format_nums3(([1])).encode()) 

    
    elif task == "moving":
        print("UR: start moving ply's")
        
        ply_number = client_socket.recv(1024).decode()
        print("UR: ply_ID is ",ply_number, "\n")

        x, y, rx, ry, error_code, cup_array =camera_data()
        print("PC: sending: x:",x,"| y:",y,"| Rz:",rx,"| Ry:",ry)
        
        client_socket.send(format_nums1((x, y, rx, ry, error_code)).encode())
        print("PC: Data send to UR")
        
        
        if error_code == 0:
            print("PC: error_code: 0 \n")
            
            data = client_socket.recv(1024).decode()
            print("UR:",data, "vacuum cups \n")
            print("PC: activating vacuüm cups:", cup_array, "\n \n")


        if error_code == 1:
            print("PC: error_code: 1 \n \n")
