import cv2 as cv
import os
import mediapipe as mp
import numpy as np
import time

def getConversionMatrix(pos1, pos2):
    Ainv = np.array([[pos1.x, pos2.x],
                 [pos1.y, pos2.y]])
    A = np.linalg.inv(Ainv)
    return A

number = 11

cTime = 0
pTime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

capture = cv.VideoCapture(0)
sample_list = []
while True:
    isTrue, frame = capture.read()
    #cv.imshow('Video', frame)
    frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_RGB)
    ## print(results.multi_hand_landmarks)
    handCoords = []
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:

            # print(f'absoluteA: {handlms.landmark[0]}')
            # print(f'absoluteB: {handlms.landmark[5]}')
            A = getConversionMatrix(handlms.landmark[0], handlms.landmark[5])
            # print(np.array(handlms.landmark[0]))
            testA = A @ np.array([handlms.landmark[0].x,handlms.landmark[0].y])
            testB = A @ np.array([handlms.landmark[5].x,handlms.landmark[5].y])
            # print(f'testA: {testA}')
            # print(f'testB: {testB}')
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
            sample_list.append(handCoords)



    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
    cv.imshow('Hands', frame)


    if cv.waitKey(20) & 0xFF == ord('d'):
        break
img = frame

capture.release()
cv.destroyAllWindows()
saveArray = np.array(sample_list)
saveArray = np.transpose(saveArray)
print(saveArray.shape)
savePath = f'./data/testData{number}'
np.save(savePath, saveArray)



