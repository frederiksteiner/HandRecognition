import cv2 as cv
import os
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class HandNetwork(nn.Module):
    def __init__(self, size_in, num_of_classes):
        super().__init__()
        self.fc0 = nn.Linear(size_in, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_of_classes)



    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x))

        return x


def getConversionMatrix(pos1, pos2):
    Ainv = np.array([[pos1.x, pos2.x],
                 [pos1.y, pos2.y]])
    A = np.linalg.inv(Ainv)
    return A

number = 0

cTime = 0
pTime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

model = torch.load('models/model2', map_location=torch.device('cpu'))
model.eval()

classes = ['L0','L1','L2','L3','L4','L5','R0','R1','R2','R3','R4','R5']
classes_tot = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]

capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    #cv.imshow('Video', frame)
    frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_RGB)
    ## print(results.multi_hand_landmarks)
    handCoords = []
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            A = getConversionMatrix(handlms.landmark[0], handlms.landmark[5])
            for id, lm in enumerate(handlms.landmark):
                newCoord = A @ np.array([lm.x, lm.y])
                newX = newCoord[0]
                newY = newCoord[1]
                handCoords.append(newX)
                handCoords.append(newY)
                h, w, c = frame.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
            mpDraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS)
        if len(handCoords) == 42:
            coordMat = np.array(handCoords)
            coordMat = np.expand_dims(coordMat, axis = 0)
            coordMat = torch.from_numpy(coordMat).float()

            out = model(coordMat)
            idx = int(np.argmax(out.detach().numpy()))
        if len(handCoords) == 84:
            coordMat = np.array(handCoords)
            firstHand = coordMat[:42]
            secondHand = coordMat[42:]
            firstHand = np.expand_dims(firstHand, axis=0)
            firstHand = torch.from_numpy(firstHand).float()
            secondHand = np.expand_dims(secondHand, axis=0)
            secondHand = torch.from_numpy(secondHand).float()
            out1 = model(firstHand)
            idx1 = int(np.argmax(out1.detach().numpy()))
            out2 = model(secondHand)
            idx2 = int(np.argmax(out2.detach().numpy()))





    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    if  len(handCoords) == 42:
        cv.putText(frame, classes[idx], (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    if len(handCoords) == 84:
        cv.putText(frame, str(int(classes_tot[idx1] + classes_tot[idx2])), (250, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    cv.imshow('Hands', frame)


    if cv.waitKey(20) & 0xFF == ord('d'):
        break
img = frame

capture.release()
cv.destroyAllWindows()
