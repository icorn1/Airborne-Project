import socket

def format_nums(integers):
    format_string = "({},{},{},{},{})".format(integers[0],integers[1],integers[2],integers[3],integers[4])
    return format_string

def close_socket():
            # Close the connection
    client_socket.close()
    server_socket.close()

def camera_data():
            #the camera gathers information. in this case some randon coordinates
    x =-400
    y =-500
    rx = 3.14
    ry = 0
    error_code =0  # Error_code 1 tells if the ply with the ply_ID given by the UR could be found or if a ply diviates more than 80% or something.
                    # Error_code 0 tells that there is no problem.
    cup_array = [0] * 24
    return x,y,rx,ry,error_code,cup_array

"""
this is part of the initialisation when starting up the robot. This needs to be run only once.
"""
communication =True
host ='192.168.0.100'        # IP address of the UR controller (PC or laptop)
port =50003                # The port used by the UR server
                            # Create a socket object and bind the socket to the host and port
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server_socket.bind((host,port))

"""
This while loop does the following:
    wait for a signal,
    accept signal,
    receive the ply_number that needs to be picked up,
    ask for the data from the CV code doing the camera stuff,
    send this data to the robot,
    waits for a response,
    if error_code = 1:
        print the error
"""
while communication:
    # open connection
    server_socket.listen(5)
    print("PC: Waiting for client connection")
    client_socket, address =server_socket.accept()
    print("PC: Connected to client \n")

    # read data from UR
    ply_number = client_socket.recv(1024).decode()
    print("UR: ply_ID is ",ply_number, "\n")

    # send data to UR from camera function
    x,y,rx,ry,error_code, cup_array =camera_data()
    print("PC: sending: x:",x,"| y:",y,"| Rz:",rx,"| Ry:",ry)
    client_socket.send(format_nums((x,y,rx,ry,error_code)).encode())
    print("PC: Data send, awaiting response. \n")

    # read data from UR
    data = client_socket.recv(1024).decode()
    print("UR: Data received",data, "\n")
    
    if error_code == 0:
        print("PC: error_code: 0 \n")
        
        data = client_socket.recv(1024).decode()
        print("UR:",data, "vacuum cups \n")
        print("PC: activate vacuüm cups:", cup_array, "\n \n")
        

    if error_code == 1:
        print("PC: error_code: 1 \n")
        
        # give two options on the GUI:
            # select another ply to fill in
            # skip this ply


