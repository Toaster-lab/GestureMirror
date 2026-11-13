# How to use the main program
To run main program 
```bash
python software/main.py
```
## Before you start
* Webcam is connected and working
* Dependencies installed from step 4 in README.md
* Optional: ESP32 and robotic hand connected and ready. The Serial connection is *commented out*! Enable Serial by removing the '#' infront of the Serial lines. Lines affected: 65, 166, 177 and 181 all need to be enabled!!
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
3. Make sure you see in the terminal that it is toggled ON or OFF.

## Adding your own training data
1. When running main, use the corresponding keys to save open / closed hands into the file: [Here is the data saved to](../software/data/tensor.pt) 
Make sure you are really using the right key for the hand state you want to save or else the model is training on wrong data!
You need to quit with pressing 'q' to actually save the tensors!!

2. To retrain the model on new data just run SAM_optimizer: [This file :)](../software/SAM_optimizer.py) You need a certain test accuracy to actually save the model, this is hard coded and can be changed on line 102!
The model parameters are saved here: [Model parameters](../software/models/modelSAM.pth)

## Useful details
* The open/closed threshold is 0.35, so it leans to open when unsure and this gave the best results when training and testing!
* The data shape of the training data should be 64: 21 landmarks * 3 dimensions (x,y,z) = 63 values plus 1 label (open / closed in binary)
* The number of hands that are tracked is 1 on purpose. This can be changed but it is untested so I would not change it!!
* You need to restart main.py to use newly trained model
* The start up pauses for 3 seconds to give time to the ESP32

## Troubleshooting
* 'No hand detected' when saving -> hand not detected in camera's view
* Camera doesn't open -> camera not available or you need to change camera index on line 55!!
* Serial error -> wrong COM or ESP32 not available
* Poor accuracy -> See: [Model limitations](model_limitations.md)