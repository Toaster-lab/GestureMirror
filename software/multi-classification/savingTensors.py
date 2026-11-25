# Pipeline: load saved tensors -> open camera -> show hand landmarks -> save tensor with label on keypress
# Keys: z=open, x=closed, c=thumb_up, v=peace, b=ok, n=point, m=three, ,=None, q = quit and write file

#dependecies:
#pretrained model used: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index#models
import torch
import cv2 as cv
import mediapipe as mp
import os
from mediapipe.tasks.python.vision import HandLandmarksConnections
from simpleTraining import HandNN

HERE = os.path.dirname(os.path.abspath(__file__))
key_to_number_label = {
    'closed' : 0,
    'open' : 1,
    'thumb_up' : 2,
    'peace' : 3,
    'ok' : 4,
    'point' : 5,
    'None' : 6
}

def write_text(text,frame) -> None:
    cv.putText(
        frame, text, (10,30),
        cv.FONT_HERSHEY_PLAIN,1,
        (22,0,255),3)
    cv.imshow
    


def saving_tensor(key, key_stroke, label, tensor, new_tensors) -> None:
    if not label in key_to_number_label:
        print('Label invalid')
    else:
        key_value = key_to_number_label[label]
        if key == ord(key_stroke):
            if tensor is None:
                print('No hand detected')
            else:
                tensor = torch.cat((tensor.flatten(), torch.tensor([key_value]))) 
                new_tensors.append(tensor)
                print(f"Saved successfully! : {label}")

def load_tensors(file_name):
    if not os.path.exists(file_name) or os.path.getsize(file_name) == 0: #Is file empty??
        example_tensor = torch.tensor([])
        print('Empty file')
        loaded_tensor = torch.reshape(example_tensor, (0,64)) #loading empty tensor with right shape
    else:
        loaded_tensor = torch.load(file_name, weights_only=True)
        print('tensors loaded')
    new_tensors = []
    return loaded_tensor, new_tensors

def saving_coordinates(result) -> torch.Tensor | None:
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
    cap = cv.VideoCapture(0)  # CAMERA INDEX, MAYBE NEED TO CHANGE
    if not cap.isOpened():
        raise RuntimeError('Cam could not open')
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options = mp.tasks.BaseOptions(model_asset_path = pretrained_model),
        num_hands = 1
    )
    return cap, options

def draw_hand(frame, result):
    height, width, _ = frame.shape
    for hand in result.hand_landmarks:
        points = []
        for landmark in hand:
            points.append((int(landmark.x * width), int(landmark.y * height)))
        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            cv.line(frame, points[connection.start],
                    points[connection.end], (0, 255, 0), 2)
        for i, (x, y) in enumerate(points):
            cv.circle(frame, (x, y), 5, (0,0,255), -1)
            cv.putText(
                frame, str(i), (x + 5, y - 5),
                cv.FONT_HERSHEY_PLAIN, 0.3,
                (255,255,255),1)

def main():
    pretrained_model_path = os.path.join(HERE, '..', 'models', 'hand_landmarker.task')
    file_name = os.path.join(HERE, 'data.pt')
    loaded_tensor, new_tensors = load_tensors(file_name)
    cap, options = start_capture(pretrained_model_path)

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv.cvtColor(frame, cv.COLOR_BGR2RGB) # needs to be RGB
                    )
                result = landmarker.detect(mp_image)
                draw_hand(frame, result)
                tensor = saving_coordinates(result)

                
                key = cv.waitKey(1)
                saving_tensor(key,'z','open',tensor,new_tensors)
                saving_tensor(key,'x','closed',tensor,new_tensors)
                saving_tensor(key,'c','thumb_up',tensor,new_tensors)
                saving_tensor(key,'v','peace',tensor,new_tensors)
                saving_tensor(key,'b','ok',tensor,new_tensors)
                saving_tensor(key,'n','point',tensor,new_tensors)
                saving_tensor(key,'m','None',tensor,new_tensors)
                if key == ord("q") or key == ord('Q'):
                    break
                cv.imshow("Hand Landmarks", frame)
                    
                    
    finally:
        cap.release()
        cv.destroyAllWindows()

        # torch.save overwrites the file, so we concatenate the old tensors with the new ones
        if new_tensors:
            saved_tensors = torch.stack(new_tensors)
            if not loaded_tensor.numel() == 0:
                loaded_tensor = loaded_tensor.reshape(-1, 64)
                saved_tensors = torch.cat((loaded_tensor, saved_tensors), dim=0)
            torch.save(saved_tensors, file_name)
            print('saved successfully')


if __name__ == "__main__":
    main()
