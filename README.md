# airborne composite automation

in this github all the codes will be given that where used to make a demo/experimental robot for airborne composite. All the codes that are given will be explained in this README.

## contact information
<details>
<summary> The students who worked on this project: </summary>

| Name:            | email:                         |
|------------------|--------------------------------|
| Jort Leroij      | jortjorisleroy@gmail.com       |
| Joël Bruinvels   | joel_bruinvels@live.nl         |
| Ixent Cornella   | icornellav@gmail.com           |
| Guillermo Forcén | G.ForcenOrdovas@student.hhs.nl |
</details>

## A Guide to the codes:

<details>
<summary> communication_code.py </summary>
  
This is a code written in python and runs on the PC used as host. This code waits for a connection from the UR. When it gets a connection it reads the ply ID send to it from the UR. Then it runs the code for the camera which analyses the ply's placed on the table and returns the real coördinates of where the ply is to be found and an error code if the ply is defect or can not be found. The PC sends this data to the UR and wait for a verification back. When received it waits for a message to activate the sucker end tool.

> This code is not completed yet and will be updated until the end of the project. The PC should also exchange communication with a PLC and Festo controller to control the vacuüm cups of the tool end.
---
</details>



<details>
<summary> communication_code.script </summary>
  
This is a code written in urscript and runs on the universal robot which is used as a client. This code moves to the start location, it sends message and a ply ID to the PC. It then waits for coördinates and error code to be send back. If received it sends a verification back that this went succesfully. then it moves the robot to the location of the ply, when done it sends a message to the PC to activate the sucker system. Then it waits a while and moves the ply to the place where the composite is to be build.

> This code is not completed yet and will be updated until the end of the project
---
</details>



<details>
<summary> Dxf_to_contour.py </summary>
  
This python script opens a dxf file that contains the 2D sketches of the ply's that are transported by the universal robot. This script converts the dxf file to png, and extracts all the contours and saves them individually. These contours can then be put into a database and be compared to the reallife ply's. This way defect ply's can be found.

> This code is not completed yet and will be updated until the end of the project
---
  
</details>
