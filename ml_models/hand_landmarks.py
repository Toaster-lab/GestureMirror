# dowloading model: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index#models
import torch
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarksConnections 

model_path = r"ml_models\models\hand_landmarker.task"

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path = model_path),
    num_hands = 2
)

file_name = r'ml_models\tensor.pt'
key_to_save = 's' #not used
cap = cv.VideoCapture(0)
loaded_tensor = torch.load(file_name, weights_only=True)
new_tensors = []

#opening cam
with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # camera filter
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # converting to mediapipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        result = landmarker.detect(mp_image)
        for hand in result.hand_landmarks:
            points = []
            for landmark in hand:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                points.append((x,y))
            for connection in HandLandmarksConnections.HAND_CONNECTIONS:
                cv.line(frame, points[connection.start],
                        points[connection.end], (0, 255, 0), 2)
            for point in points:
                cv.circle(frame, point, 5, (0,0,255), -1)

        height, width, _ = frame.shape

        for hand in result.hand_landmarks:
            for i, point in enumerate(hand):
                x = int(point.x * width)
                y = int(point.y * height)

                cv.circle(frame, (x,y), 5, (0,0,255), -1)
                cv.putText(
                    frame, str(i), (x + 5, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255,255,255),1
                )

        for hand in result.hand_landmarks:
            coordiante = []
            for landmark in hand:
                coordiante.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])
            tensor = torch.tensor(coordiante, dtype=torch.float32)

        cv.imshow("hand marks", frame)
        key = cv.waitKey(1)
        if key ==ord('x'):
            tensor = tensor.flatten()
            tensor = torch.cat((tensor, torch.tensor([1.0]))) # we are adding a one, so we can say it f. e. a closed hand 
            new_tensors.append(tensor)
            print("Saved successfully! : open hand")
            print(tensor.shape) # expected: 64 - true
        if key ==ord('z'):
            tensor = tensor.flatten()
            tensor = torch.cat((tensor, torch.tensor([0.0]))) # we are adding a one, so we can say it f. e. a closed hand 
            new_tensors.append(tensor)
            print("Saved successfully! : closed hand")
            print(tensor.shape) # expected: 64 - true
        if key == ord("q"):
            break
        

cap.release()
cv.destroyAllWindows()

# Problem: torch.save overwrites file, so we need to load the saved tensors and add the new ones to them!
if new_tensors:
    saved_tensors = torch.stack(new_tensors)

    if not loaded_tensor.numel() == 0:
        loaded_tensor = loaded_tensor.reshape(-1, 64)
        saved_tensors = torch.cat((loaded_tensor, saved_tensors), dim=0)

    torch.save(saved_tensors, file_name)
    print('saved successfully')