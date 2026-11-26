#dependecies:
#pretrained model used: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index#models
import sys
from pathlib import Path
import torch
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarksConnections
# SAM_optimizer.py lives one folder up, in software/
sys.path.append(str(Path(__file__).resolve().parents[1]))
from SAM_optimizer import SAMClassification

MEIMEI_WINDOW = 'Mei Mei says HII!!!'


def load_model(model_path):
    model = SAMClassification()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print('loaded succesfully')
    return model

def get_prediction(model, tensor, frame):
    if not tensor.shape == (1, 63):
        raise AssertionError('Tensor wrong shape')
    with torch.no_grad():
        percent = model(tensor).sigmoid()
    # I want it rather when it is unsure to be open then closed
    hand_open = bool(percent > 0.35)
    cv.putText(frame, "open" if hand_open else "closed", (20,40),
        cv.FONT_HERSHEY_PLAIN,1,
        (0,0,255),3)
    return hand_open

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
    cap = cv.VideoCapture(0)  # CAMERA INDEX, MAYBE NEED TO CHANGE
    if not cap.isOpened():
        raise RuntimeError('Cam could not open')
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options = mp.tasks.BaseOptions(model_asset_path = pretrained_model),
        num_hands = 1 #changed to one because it bugged before and for my hand I only need one
    )
    return cap, options

def getting_results(cap, landmarker):
    ret, frame = cap.read()
    if not ret:
        return None, None, True # break
    # camera filter
    rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # needs to be RGB

    # converting to mediapipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
        )
    result = landmarker.detect(mp_image)
    return result, frame, False

def draw_hand(frame, result):
    height, width, _ = frame.shape

    for hand_index, hand in enumerate(result.hand_landmarks):
        handeness = result.handedness[hand_index][0]
        handeness_string = str(handeness.category_name).lower()
        cv.putText(
            frame, handeness_string, (20, 60),
            cv.FONT_HERSHEY_PLAIN,1,
            (22,0,255),3)

        points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in hand]
        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            cv.line(frame, points[connection.start],
                    points[connection.end], (0, 255, 0), 2)
        for i, (x, y) in enumerate(points):
            cv.circle(frame, (x,y), 5, (0,0,255), -1)
            cv.putText(
                frame, str(i), (x + 5, y - 5),
                cv.FONT_HERSHEY_PLAIN, 0.3,
                (255,255,255),1)

def show_meimei(image, hand_open, window_open):
    if hand_open:
        cv.imshow(MEIMEI_WINDOW, image)
    elif window_open:
        cv.destroyWindow(MEIMEI_WINDOW) # crashes if the window doesn't exist
    return hand_open

def main():
    #loading paths
    model_path = r'software\models\modelSAM.pth'
    pretrained_model_path = r'software\models\hand_landmarker.task'
    meimei = cv.imread(r'docs\images\meimei.png')
    if meimei is None:
        raise FileNotFoundError('Could not load meimei.png')
    window_open = False
    # loading model
    model = load_model(model_path=model_path)
    # opening capture and defining config for pretrained model
    cap, options = start_capture(pretrained_model_path)

    #opening cam
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                result, frame, break_ = getting_results(cap, landmarker)
                if break_:
                    break
                draw_hand(frame, result)

                tensor = saving_coordinates(result)
                # False if no hand is visible
                hand_open = tensor is not None and get_prediction(model, tensor, frame)
                window_open = show_meimei(meimei, hand_open, window_open)

                cv.imshow("Hand Classification", frame)
                key = cv.waitKey(1)
                if key == ord('q') or key == ord('Q'):
                    break

    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
