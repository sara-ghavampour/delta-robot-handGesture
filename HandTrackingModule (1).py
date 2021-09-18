import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detection_con, self.tracking_con)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, image, draw=False):

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return image

    def find_position(self, image, hand_number=0, draw=False):
        landmarks_list = []

        if self.results.multi_hand_landmarks:
            numbered_hand = self.results.multi_hand_landmarks[hand_number]
            for lm_id, lm in enumerate(numbered_hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append([lm_id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return landmarks_list


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, flipped_image = cap.read()
        image = cv2.flip(flipped_image, 1)

        image = detector.find_hands(image)

        landmarks_list = detector.find_position(image, draw=False)
        if len(landmarks_list) != 0:
            print(landmarks_list[2])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, "fps: " + str(int(fps)), (5, 30), cv2.FONT_ITALIC, 1, (128, 0, 0), 1)

        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
