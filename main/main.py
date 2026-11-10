#dependecies: 
#pretrained model used: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index#models
import torch
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarksConnections 

#initalizing Serial:
port : str = "COM9"
import serial
import time
#esp = serial.Serial(port, 115200, timeout=2)
time.sleep(5) 
print("Serial opened")
roboticHand_open : bool = True
mirroring_gesture : bool = False
personsHand_open = None

#initalizing model:
from SAM_optimizer import SAMClassification
model_path = r"main\modelSAM.pth"
model = SAMClassification()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
print('loaded succesfully')

#Paths
model_path = r'main\models\hand_landmarker.task'
file_name = r'main\tensor.pt'

#Tensors
loaded_tensor = torch.load(file_name, weights_only=True)
new_tensors = []

cap = cv.VideoCapture(0)
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path = model_path),
    num_hands = 1 #changed to one because it bugged before and for my hand I only need one
)

#opening cam
with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # camera filter
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # needs to be RGB

        # converting to mediapipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        result = landmarker.detect(mp_image)
        for hand_index, hand in enumerate(result.hand_landmarks):
            handeness = result.handedness[hand_index][0]
            handeness_string = str(handeness.category_name).lower()
            cv.putText(
                frame, handeness_string, (20, 60),
                cv.FONT_HERSHEY_PLAIN,1,
                                (22,0,255),3)
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
                    cv.FONT_HERSHEY_PLAIN, 0.3,
                    (255,255,255),1)

        tensor  = None 
        for hand in result.hand_landmarks:
            coordiante = []
            for landmark in hand:
                coordiante.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])
            tensor = torch.tensor(coordiante, dtype=torch.float32)
            tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                predicition = model(tensor)
                percent = predicition.sigmoid()
                if percent <= 0.35:  # I want it rather when it is unsure to be open then closed
                    percent = "closed"
                    personsHand_open : bool = False
                else:
                    percent = "open"
                    personsHand_open : bool = True
                cv.putText(frame, percent, (20,40),
                cv.FONT_HERSHEY_PLAIN,1,
                (0,0,255),3)

        cv.imshow("Hand Classification", frame)
        key = cv.waitKey(1)
        if key ==ord('x') or key == ord('X'):
            tensor = tensor.flatten()
            tensor = torch.cat((tensor, torch.tensor([1.0]))) # we are adding a one, so we can say it f. e. a open hand 
            new_tensors.append(tensor)
            print("Saved successfully! : open hand")
            print(tensor.shape) # expected: 64 - true - this is just for making sure everything aright
        if key ==ord('z') or key==ord('Z'):
            tensor = tensor.flatten()
            tensor = torch.cat((tensor, torch.tensor([0.0]))) # we are adding a zero, so we can say it f. e. a closed hand 
            new_tensors.append(tensor)
            print("Saved successfully! : closed hand")
            print(tensor.shape) # expected: 64 - true
        if key == ord("q") or key==ord('Q'):
            #esp.write(b'OPEN\n')
            print('sent open') 
            personsHand_open = True
            roboticHand_open = personsHand_open
            break
        if key == ord("c") or key==ord("C"):
            mirroring_gesture = not mirroring_gesture
        if mirroring_gesture == True:
            if personsHand_open != roboticHand_open:
                if personsHand_open == True:
                    #esp.write(b'OPEN\n')
                    print('sent open')  
                    roboticHand_open = personsHand_open                  
                elif personsHand_open == False:
                    #esp.write(b'CLOSED\n')
                    print('sent closed')
                    roboticHand_open = personsHand_open
                else:
                    pass

cap.release()
cv.destroyAllWindows()

# Problem: torch.save overwrites file, so we need to load the saved tensors and add the new ones to them!
#Solution: We load the old tensor, stack the new ones together and then merge them using 'torch.cat'
if new_tensors:
    saved_tensors = torch.stack(new_tensors)

    if not loaded_tensor.numel() == 0:
        loaded_tensor = loaded_tensor.reshape(-1, 64) #making it right size
        saved_tensors = torch.cat((loaded_tensor, saved_tensors), dim=0) # dim=0 -> new row

    torch.save(saved_tensors, file_name)
    print('saved successfully')