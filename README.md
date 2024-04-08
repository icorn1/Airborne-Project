# Airborne Composite Automation

In this github, we aim to share the code for the project we (students from THUAS) developed for a demonstration/experimental Universal Robot. The robot mimics an actual industrial robot from Airborne's facilities (which you can see it with more detail here: https://youtu.be/NmRDtkKQZAs), that way the company can use it for divulgation and demonstration puroposes at fairs and events, and expermiental testing of grippers or End Effectors on a smaller robot instead of the actual industrial robot. 


## Contact Information
<details>
<summary> The students who worked on this project are: </summary>
<br />
      
| Name:            | email:                         |
|------------------|--------------------------------|
| Jort Leroij      | jortjorisleroy@gmail.com       |
| Joël Bruinvels   | joel_bruinvels@live.nl         |
| Ixent Cornella   | icornellav@gmail.com           |
| Guillermo Forcén | G.ForcenOrdovas@student.hhs.nl |
</details>

## A Guide to the codes:

<details><summary><b>Explanation of python code</b></summary>
      <br />
      <details><summary><i>main_communication_code.py</i></summary>
            
This is the main code that is run in python on the PC. The PC is the host and the Universal Robot (UR) and PLC are the clients. This code is the main code and is simple to use: Download all the python codes onto the PC and the URScripts onto the robot and run the code on the PC. Than for the rest everything can be done from the GUI on the robot.

In this code all the settings are given and can be addapted (such as the detection threshhold for instance). An explanation for all the variables is given in the code itself. When the code is executed the PC opens a socket connection and waits till it is accepted by the UR. When the UR is started in the GUI you can select "start calibration" or "start moving ply's". When you select "start calibration", this will be send to the PC and the PC will make sure to run the right code for the calibration and will preform the calibration automaticaly (the only thing you will have to do is replace the tool end for the calibration tool). When you select "start moving ply's" the PC will be notified and will run the right script for this. Again the rest is all preformed automaticaly, unless there is a defect ply. When this happens, in the GUI, you can chose to skip this ply or to chose another ply to fill in in the composite.

> This code is not completed yet and will be updated until the end of the project.
---
</details>
      <details><summary><i>Dxf_to_contour.py</i></summary>

This python script opens a dxf file that contains the 2D sketches of the ply's that are transported by the universal robot. This script converts the dxf file to png, and extracts all the contours and saves them individually. These contours can then be put into a database and be compared to the reallife ply's. This way defect ply's can be found.

> This code is not completed yet and will be updated until the end of the project
---
</details>
      <details><summary><i>contours.py</i></summary>
  
This python script accounts for detecting contours. The GUI sends a contour to the laptop which needs to be found in the camera image. This script contains the functions necessary for comparing contours which will allow for the detection of the right contour.

> This code is not completed yet and will be updated until the end of the project
---
</details>
      <details><summary><i>machine_vision_functions.py</i></summary>

...

> This code is not completed yet and will be updated until the end of the project
---
</details>
      <details><summary><i>calibrate.py</i></summary>
  
This file is used for storing the calibration functions. These functions allow for the calibration of the camera to the universal robot. Required for its use are the images and robot poses captured during the calibration. The function returns a translation matrix which can be used for converting camera coordinates to their corresponding robot pose. 

> This code is not completed yet and will be updated until the end of the project
---
</details>
      <details><summary><i>PLC_communication.py</i></summary>
  
This file is used for storing the function for communicating to the PLC.

> This code is not completed yet and will be updated until the end of the project
---
</details>
</details>
<br />
<details><summary><b>Explanation of URScript code</b></summary>
      <br />
      <details><summary><i>start_moving_ply.script</i></summary>
  
This is a code written in urscript and runs on the universal robot which is used as a client. This code is activated when the ply's needs to be moved and a composite needs to be made. The current version of this code works together with the PC to find 1 single ply, it does not matter which ply, pick this ply up and place it on another location. This code is executed when on the GUI of the robot the button "start moving ply's" is pressed.

> This code is not completed yet and will be updated until the end of the project
---
</details>
      <details><summary><i>calibration.script</i></summary>
  
This code is written in urscript and runs on the UR. This code communicates with the PC and preforms the calibration automaticaly. All that needs to be done is that the tool end needs to be manually replaced with the calibration tool. This code is executed when on the GUI of the robot the button "start calibration" is pressed. After executing this code the robot will move to a few locations and will pause at every location for the camera to take a picture. The picture and the pose are then compared on the PC and the calibration is finished.

> This code is not completed yet and will be updated until the end of the project
---
</details>
      <details><summary><i>Initialise.script</i></summary>
  
This code is written in urscript and runs on the UR. This code initialises the variables such as starting position, safe height for the robot, the host ID, the port and the ply ID and turns on the digital output pins that are used..

> This code is not completed yet and will be updated until the end of the project
---
</details>
      <details><summary><i>De_initialise.script</i></summary>
  
This code is written in urscript and runs on the UR. This code turns off all of the digital output pins.

</details>
</details>
<br />
<details><summary><b>Explanation of URCaps code</b></summary>
      <br />
	URCaps code is based on Java, with Swing for the application in this case. This way we can create custom installation and program modules for UR cobots that do exactly what we want, and they can also act as a GUI.
 What our URCap does exactly (demonstration) is provide a GUI and a simple integration for a Cobot to a laptop, to perform the logic of the robot. For it to be automated, the first step is to press the big green button "START". 

## Contents of the GUI:
- Company (Airborne) logo + student credits.
- Big green button to start the program.
- Settings boxes to select wheter user wants to do calibration or run a certain laminate.
- Corresponding laminate image.

## How to compile and install the URCap
You'll need a Linux based enviornment, with maven installed. Then, enter the URCaps Code folder, and from there run "mvn install -Premote" if you want to install it on to the robot, or "mvn install -P ursimvm" if you plan to do it on the URSimulator. <b>Keep in mind, you might need to adjust the IP addresses on the POM file! </b> You can find the full list of dependencies on the development guide below.

I strongly suggest to check the development guide for URCaps if you want to develop your own URCap.
Development guide:
https://plus.universal-robots.com/media/1810567/urcap_tutorial_swing.pdf
Style guide:
https://www.universal-robots.com/media/1802558/urcaps_style_guide-v10.pdf

## Credits
This URCap uses source code from another URCap, credits to https://github.com/BomMadsen/URCap-ScriptCommunicator.
</details>
