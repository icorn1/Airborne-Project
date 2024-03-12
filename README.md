# airborne composite automation

in this github all the codes will be given that where used to make a demo/experimental robot for airborne composite. All the codes that are given will be explained in this README.

## contact information
<details>
<summary> The students who worked on this project: </summary>

| Name:            |
|------------------|
| Jort Leroij      |
| Joël Bruinvels   |
| Ixent Cornella   |
| Guillermo Forcén |

</details>

## A Guide to the codes:

<details>
<summary> Python communication code </summary>
  
This is a code written in python and runs on the PC used as host. This code waits for a connection from the UR. When it gets a connection it reads the ply ID send to it from the UR. Then it runs the code for the camera which analyses the ply's placed on the table and returns the real coördinates of where the ply is to be found and an error code if the ply is defect or can not be found. The PC sends this data to the UR and wait for a verification back. When received it waits for a message to activate the sucker end tool.

> This code is not completed yet and will be updated until the end of the project. The PC should also exchange communication with a PLC and Festo controller to control the vacuüm cups of the tool end.
---

</details>

<details>
<summary> Urscript communication code </summary>
  
This is a code written in urscript and runs on the universal robot which is used as a client. This code moves to the start location, it sends message and a ply ID to the PC. It then waits for coördinates and error code to be send back. If received it sends a verification back that this went succesfully. then it moves the robot to the location of the ply, when done it sends a message to the PC to activate the sucker system. Then it waits a while and moves the ply to the place where the composite is to be build.

> This code is not completed yet and will be updated until the end of the project
---
  
</details>
