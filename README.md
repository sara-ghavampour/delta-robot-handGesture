# HandGesture
This code let the user to control Delta robot with hand motions.
It uses OpenCV and MediaPipe.
The main module is Movements Module.
It uses these [landmarks](https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn-images-1.medium.com%2Ffit%2Ft%2F1600%2F480%2F1*WhYiJkSaqJAMEloRIiWHTQ.png&tbnid=UPkP4H6khoymMM&vet=12ahUKEwignsWP76uBAxX2gv0HHZXGCTgQMygRegQIARB8..i&imgrefurl=https%3A%2F%2Ftowardsdatascience.com%2Fexquisite-hand-and-finger-tracking-in-web-browsers-with-mediapipes-machine-learning-models-2c4c2beee5df&docid=xu48bmGzyvZQxM&w=1600&h=480&q=mediapipe%20hand%20landmarks&client=ubuntu-sn&ved=2ahUKEwignsWP76uBAxX2gv0HHZXGCTgQMygRegQIARB8) to specify hand moves.
For example we can track of labdmark with id of 0 and check the position of landmark 0. if the x cordinate of this landmark increase means hand is moving up.
