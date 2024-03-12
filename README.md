# airborne composite automation

in this github all the codes will be given that where used to make a project using a universal robot work.

<details>
<summary> The students who worked on this project: </summary>

| Name:            |
|------------------|
| Jort Leroij      |
| Joël Bruinvels   |
| Ixent Cornella   |
| Guillermo Forcén |

</details>

All the programming was done in Python, Urscript and Tiaportal. In this README every code will be explained.

## Python communication code

<details>
## <summary> Urscript communication code </summary>
This is a code written in urscript and run onto the universal robot. This code moves to the start location, it sends message and a ply ID to the laptop. It then waits for coördinates and error code to be send back. If received it sends a verification back that this went succesfully. then it moves the robot to the location of the ply, when done it sends a message to the PC to activate the sucker system. Then it waits a while and moves the ply to the place where the composite is to be build.
</details>
