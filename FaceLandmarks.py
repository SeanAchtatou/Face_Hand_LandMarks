
import mediapipe as mp
import cv2


class FaceLandMarks():

    def __init__(self, num_face=1, minConfDet = 0.5, minConfTrack = 0.5):
        self.mpDraw = mp.solutions.drawing_utils
        mpFaceLandmark = mp.solutions.face_mesh
        self.faceMesh = mpFaceLandmark.FaceMesh(max_num_faces=num_face)

    def run(self,image):
        result = self.faceMesh.process(image)
        if result.multi_face_landmarks:
            for facesLm in result.multi_face_landmarks:
                for id,lm in enumerate(facesLm.landmark):
                    imy,imx,imh = image.shape
                    x_value = lm.x*imx
                    y_value = lm.y*imy
                    cv2.putText(image, ".", (int(x_value),int(y_value)), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),1)

        return image



if __name__ == "__main__":
    fL = FaceLandMarks()


