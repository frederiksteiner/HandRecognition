# HandRecognition
Little computer vision project, where the number of fingers are counted. 

The goal of this project was to count the number of fingers held up in the air. If only one hand is up in the air, it should show if it is the right or left hand and how many fingers are up in the air. To do this, I was working with mediapipe and the x- and y-coordinates of the 21 different handlandmarks. This 42 long vector of coordinates can then be plugged into a neural network to do the classification task.


