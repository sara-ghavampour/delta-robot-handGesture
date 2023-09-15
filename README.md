# HandGesture
This code let the user to control Delta robot with hand motions.
It uses OpenCV and MediaPipe.
The main module is Movements Module.
It uses these [landmark1](https://cdn-images-1.medium.com/fit/t/1600/480/1*WhYiJkSaqJAMEloRIiWHTQ.png) ,[landmark2](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7gKTpz0UV9CvuZMmy_B5ITWxeLGhlikolT_nh5Sxp8D34fX_-giYsEylY2unrJQRs2eY&usqp=CAU) to specify hand moves.
For example we can track of labdmark with id of 0 and check the position of landmark 0. if the x cordinate of this landmark increase means hand is moving up.
