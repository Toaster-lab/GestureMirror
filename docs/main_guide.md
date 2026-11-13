# How to use the main program

## Before you start
* Webcam is connected and working
* Dependencies installed from step 4
* Optional: ESP32 and robotic hand connected and ready. The Serial connection is *commented out*, make sure to enable them before running! Lines affected: 65, 166, 177 and 181 all need to be changed!!
* Set the right Serial port on line 127

## What you see
* Hand skeleton: MediaPipe model sets landmarks
* Text on the top left: Handedness (left / right) and predicted state (open / closed)
* Terminal: What hand position is sent to ESP32 and if mirroring is toggled

## Controls
```
Key | Action
-------------------
C   | Toggling mirroring on / off
X   | Save current landmarks as open for training
Z   | Save current landmarks as closed for training
Q   | Quit camera and resets hand to open
```

## How to mirror robotic hand
1. Delete lines that comment out Serial, see *Before you start* for specific lines
2. Press C to toggle on mirroring 
3. Make sure you see in the terminal that it is toggeled ON or OFF.

## Adding your own training data
1. When running main, use the corresponding keys to save open / closed hands into the file: [Here is the data saved to](software\data\tensor.pt) 
Make sure you are really using the right key for the hand state you want to save!

2. To retrain the model just run SAM_optimizer: [This file :)](software\SAM_optimizer.py) 
The model parameters are saved here: [Model parameters](software\models\modelSAM.pth)

If you want to train the model fresh, just empty the hand training data and model parameters. Do NOT empty the MediaPipe model: [Do NOT delete](software\models\hand_landmarker.task)

## Troubleshooting
* 'No hand detected' when saving -> hand not detected in camera's view
* Camera doesn't open -> camera not available or you need to change camera index on line 55!!
* Serial error -> wrong COM or ESP32 not available
* Poor accuracy -> See: [Model limitations](docs\model_limitations.md)