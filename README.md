Retro PiCam: Dedicated Digital Camera Project (WIP)
This project is a work-in-progress aimed at creating a dedicated, single-purpose digital camera system running on the Raspberry Pi. The goal is to replicate the focused, tactile shooting experience of an old-school digital camera while leveraging the high-quality raw sensor data from the Arducam IMX335 module.
The system will eventually be streamlined to boot directly into the application and utilize a Waveshare 640x480 DSI display for the viewfinder.
________________________________________
Project By
Sean Burrage
________________________________________
Overview: Dedicated Raw Capture System
This repository contains two core components:
1.	The Camera App (imx335.py): Provides the live preview, interactive controls, and manages high-speed capture of either processed JPEGs or raw sensor data.
2.	The Processing Utility (process_raw.py): Converts the hardware-specific raw files into universally viewable images, ensuring data integrity is maintained.
Files
# File	Description
imx335.py	The main camera application for live preview and capture control.
process_raw.py	A utility script to convert the raw .npz files into viewable 16-bit PNG images.
________________________________________  

## Setup and Dependencies  
This project requires the standard Raspberry Pi camera software stack (libcamera) and several Python libraries.

Hardware Prerequisites  

•	Raspberry Pi CM5 or CM4
•	Arducam IMX335 Camera Module  
•	(Target Display: Waveshare 640x480 DSI)  

  
Software Installation
Ensure your system is up-to-date and the camera is detected:  
``` 
sudo apt update  
sudo apt upgrade -y  
libcamera-hello --list-cameras
```

Install the required Python libraries:
```
sudo apt install -y python3-picamera2 python3-opencv python3-numpy
```
________________________________________
## Using the Camera Application (imx335.py)
The main application provides a live preview and interactive controls for capture.  
### Operation
1.	Run the script from your terminal:
```
python3 imx335.py
```
2.	A preview window opens (currently targeting the 640x480 resolution intended for the DSI display). Use the following keys for control:
Key	Function
```
Spacebar / S	Capture still image (based on current mode).
H  Toggle hints
R  Toggle capture mode between JPEG (processed) and RAW (packed .npz).
M  Toggle Auto Exposure (AE) / Manual Exposure mode.
[ / ]  Adjust Exposure Time (in Manual mode).
- / =  Adjust Analogue Gain (in Manual mode).
Q  Quit the application.
```
## Capture Modes  
### Mode	Format & File	Output Location	Notes  
  
**JPEG	RGB888**  
.jpg  
/home/user/Pictures  
Standard processed image.  
Utilizes the Image Signal Processor (ISP).  
  
**RAW	SRGGB10_CSI2P**  
.npz  
/home/user/Pictures  
10-bit raw Bayer data.  
Captures what the sensor sees directly.  
Requires external processing.  
________________________________________
##  Post-Processing Raw Files (process_raw.py)  
The .npz files contain raw, 10-bit packed Bayer data specific to the Raspberry Pi's CSI-2 pipeline. They cannot be viewed directly. The process_raw.py script unpacks, debayers, and converts this data into viewable, high-quality 16-bit PNG files.  
### Why Processing is Necessary  
The IMX335's 10-bit data is physically packed (four 10-bit pixels are compressed into five bytes) to optimize memory bandwidth. The process_raw.py script reverses this complex hardware packing, removes any row padding bytes, and then uses OpenCV to perform the demosaicing (debayering) and color correction.  
### Usage
1.	Ensure all your .npz files are in the defined INPUT_DIR (default: /home/user/Pictures).
2.	Run the processing script:
```
python3 process_raw.py
```
3.	Processed 16-bit PNG images (with full color depth) will be saved to the OUTPUT_DIR (default: /home/user/Pictures/Processed).
   
Configuration (Inside process_raw.py)
You can adjust these settings within the script file for different results:
Variable	Default Value	Description
INPUT_DIR	/home/user/Pictures	Location of the source .npz files.
OUTPUT_DIR	/home/user/Pictures/Processed	Destination for the output .png files.
BAYER_PATTERN	cv2.COLOR_BayerRG2RGB	OpenCV debayer pattern. Adjust this if colors appear incorrect (e.g., green tint).
ENABLE_GAMMA	True	Applies sRGB gamma correction for standard screen viewing. Set to False for linear, scientific data.

