# dowloading model: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index#models

import cv2 as cv
import mediapipe as mp

model_path = "ml_models/models/hand_landmarker.task"

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options = mp.tasks.BaseOptions(model_asset_path = model_path),
    num_hands = 2
)

cap = cv.VideoCapture(0)
    
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

        height, width, _ = frame.shape

        for hand in result.hand_landmarks:
            for i, point in enumerate(hand):
                x = int(point.x * width)
                y = int(point.y * height)

                cv.circle(frame, (x,y), 5, (0,255,0), -1)
                cv.putText(
                    frame, str(i), (x + 5, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255,255,255),1
                )
        cv.imshow("hand marks", frame)

        if cv.waitKey(1) == ord("q"):
            break

cap.release()
cv.destroyAllWindows()