# virtual-mouse-controller
This script can let user control their mouse cursor and click using hand through webcam
It uses Mediapipe hand model for tracking hand, and we have a particular bounding region which work as a trackpad  so if user take the finger inside that region it will start moving cursor and wherever user go cursor will follow the same 

Its concept is quite simple,  it can come in handy one user get used to this, 

Required libraries:

* PyAutoGUI 0.9.53
* Mediapipe 0.8.7.3
* OpenCV 4.5.3 or Higher

To start it:
'''
python controller.py
'''
