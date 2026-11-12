# Limitations

## Model
* Distance and position: Accuracy drops as the hand moves away from the camera. The model uses raw landmark coordinates, which change with hand size and position in the frame. Fix: normalize landmarks relative to the wrist and hand size.
* Handedness: Training data doesn't account for left vs. right hands. Fix: mirror left-hand landmarks or add handedness as a feature.
* Two classes only: It only distinguishes open from closed. Intermediate or other gestures are forced into one of the two.
* Limited data: Training data is not that much and also only comes from camera so may not be perfect for other people.

## Input quality
* Lighting and contrast: MediaPipe needs a clear view of the hand to produce good landmarks. The classifier depends on this.
* Camera quality: A low-quality camera gives noisy landmarks and less accurate predictions.

## Usability
* Startup time: Loading takes a few seconds 
* Setting up new training data: tensor.pt must contain an empty tensor of the right shape before collecting data.
* Single hand: Only one hand is tracked at a time.
