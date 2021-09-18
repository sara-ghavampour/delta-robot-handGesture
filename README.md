# HandGesture
This code let the user to control Delta robot with hand motions.
It uses OpenCv and mediaPipe.
main module is HandTrackingModule.
it uses these https://google.github.io/mediapipe/images/mobile/hand_landmarks.png  landmarks to specify hand moves.
for example we can track of labdmark with id of 0 and check the position of landmark 0. if the x cordinate of this landmark increase means hand is moving up.
