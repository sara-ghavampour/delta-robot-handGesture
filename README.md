# HandGesture
This code let the user to control Delta robot with hand motions.
It uses OpenCV and MediaPipe.
The main module is Movements Module.
It uses these[landmarks] (https://google.github.io/mediapipe/images/mobile/hand_landmarks.png) to specify hand moves.
For example we can track of labdmark with id of 0 and check the position of landmark 0. if the x cordinate of this landmark increase means hand is moving up.
