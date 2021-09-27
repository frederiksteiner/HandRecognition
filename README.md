# HandRecognition
Little computer vision project, where the number of fingers are counted. 

The goal of this project was to count the number of fingers held up in the air. If only one hand is up in the air, it should show if it is the right or left hand and how many fingers are up in the air. To do this, I was working with mediapipe and the x- and y-coordinates of the 21 different handlandmarks. This 42 long vector of coordinates can then be plugged into a neural network to do the classification task.

The hands.py is used to generate the data with the help of mediapipe and the computerwebcam
The modelTrainer.py is used to train and test the neural Network for the classification task
The handNumberTest.py is the program, where one can test the implementation with the help of the webcam.


