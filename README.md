# HandRecognition
Little computer vision project, where the number of fingers are counted. 

The goal of this project was to count the number of fingers held up in the air. If only one hand is up in the air, it should show if it is the right or left hand and how many fingers are up in the air. To do this, I was working with mediapipe and the x- and y-coordinates of the 21 different handlandmarks. This 42 long vector of coordinates can then be plugged into a neural network to do the classification task.

The hands.py is used to generate the data with the help of mediapipe and the computerwebcam
The modelTrainer.py is used to train and test the neural Network for the classification task
The handNumberTest.py is the program, where one can test the implementation with the help of the webcam.

## Examples of Recognition

![R1](https://user-images.githubusercontent.com/56148594/134936016-9dabaeaf-0efe-49a4-8628-01ad42a1d7b6.png)

Here the green text stands for Right Hand and one finger.

![L3](https://user-images.githubusercontent.com/56148594/134936399-df40632b-34b7-4f4e-ac8c-53a7e0d77d6c.png)

L3 stands for Left Hand and 3 fingers

![10](https://user-images.githubusercontent.com/56148594/134936445-47368b78-703e-4ad8-8cc5-38df6fd33a34.png)

Here the 10 stands for the total number of fingers held up.

## Network architecture

