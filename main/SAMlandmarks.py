#dependecies:
#pretrained model used: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index#models
import torch
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarksConnections
from SAM_optimizer import SAMClassification

def load_paths(model_path, model, pretrained_model, file_name):
    return model_path, model, pretrained_model, file_name

def load_model(model_path):
    model = SAMClassification()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print('loaded succesfully')
    return model

def load_tensors(file_name):
    loaded_tensor = torch.load(file_name, weights_only=True)
    new_tensors = []
    return loaded_tensor, new_tensors
def get_prediction(model, tensor, frame):
    with torch.no_grad():
        predicition = model(tensor)
        percent = predicition.sigmoid()
        if percent <= 0.35:  # I want it rather when it is unsure to be open then closed
            percent = "closed"
            hand_state = 0
        else:
            percent = "open"
            hand_state = 1
        cv.putText(frame, percent, (20,40),
            cv.FONT_HERSHEY_PLAIN,1,
            (0,0,255),3)
        return hand_state
def saving_coordinates(result):
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
    return tensor
def start_capture(pretrained_model):
    cap = cv.VideoCapture(0)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options = mp.tasks.BaseOptions(model_asset_path = pretrained_model),
        num_hands = 1 #changed to one because it bugged before and for my hand I only need one
    )  
    return cap, options
def main():
    #loading paths
    model_path, model, pretrained_model_path, file_name = load_paths(r'main\modelSAM.pth',r'main\modelSAM.pth',r'main\models\hand_landmarker.task',r'main\tensor.pt')
    # loading model
    model = load_model(model_path=model_path)
    # loading tensors
    loaded_tensor, new_tensors = load_tensors(file_name)
    # opening capture and defining config for pretrained model
    cap, options = start_capture(pretrained_model_path)

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
                    
            tensor = saving_coordinates(result)
            if tensor is not None:
                hand_state = get_prediction(model, tensor, frame)

            cv.imshow("Hand Classification", frame)
            key = cv.waitKey(1)
            if key ==ord('x') or key == ord('X'):
                if tensor is None:
                    print('No hand detected')
                else: 
                    tensor = tensor.flatten()
                    tensor = torch.cat((tensor, torch.tensor([1.0]))) # we are adding a one, so we can say it f. e. a open hand
                    new_tensors.append(tensor)
                    print("Saved successfully! : open hand")
                    print(tensor.shape) # expected: 64 - true - this is just for making sure everything aright
            if key ==ord('z') or key==ord('Z'):
                if tensor is None:
                    print('No hand detected')
                else: 
                    tensor = tensor.flatten()
                    tensor = torch.cat((tensor, torch.tensor([0.0]))) # we are adding a zero, so we can say it f. e. a closed hand
                    new_tensors.append(tensor)
                    print("Saved successfully! : closed hand")
                    print(tensor.shape) # expected: 64 - true
            if key == ord("q") or key==ord('Q'):
                break

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


if __name__ == "__main__":
    main()
