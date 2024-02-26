
import mediapipe as mp
import cv2


class HandLandMarks():

    def __init__(self, num_hands=1, minConfDet = 0.3, minConfTrack = 0.3):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHandLandmark = mp.solutions.hands
        self.HandMesh = self.mpHandLandmark.Hands(max_num_hands=num_hands,min_detection_confidence=minConfDet)

    def run(self,image):
        result = self.HandMesh.process(image)
        if result.multi_hand_landmarks:
            for handLm in result.multi_hand_landmarks:
                for id,lm in enumerate(handLm.landmark):
                    imy,imx,imh = image.shape
                    x_value = lm.x*imx
                    y_value = lm.y*imy
                    cv2.circle(image, (int(x_value),int(y_value)), 3, (0,255,0),cv2.FILLED)
                self.mpDraw.draw_landmarks(image,handLm, self.mpHandLandmark.HAND_CONNECTIONS)

        return image



if __name__ == "__main__":
    fL = HandLandMarks()


