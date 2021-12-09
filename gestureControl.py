# Credit to TechVidvan. I am reimplementing and adding my own code to work with gesture control, but I am not attempting to pass off all related files, specifically the data files, as entirely my own work. Credit to TechVidvan, especially for the data files for the hand gestures.

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
#import time

def main():
    fileHands = mp.solutions.hands
    hands = fileHands.Hands(max_num_hands=1, min_detection_confidence=0.98)
    drawHands = mp.solutions.drawing_utils
    gestureRecognizer = load_model("mp_hand_gesture")
    file = open("gesture.names", "r")
    gestureNames = file.read().split('\n')
    file.close()
    vid = cv.VideoCapture(0)
    while True:
        _, frame = vid.read()
        x, y, x = frame.shape
        frame = cv.flip(frame, 1)
        frameColored = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gesturePrediction = hands.process(frameColored)
        gestureName = ""
        if (gesturePrediction.multi_hand_landmarks):
            handLandmarks = []
            for handDetails in gesturePrediction.multi_hand_landmarks:
                for landmark in handDetails.landmark:
                    landmarkX = int(landmark.x * x)
                    landmarkY = int(landmark.y * y)
                    handLandmarks.append([landmarkX, landmarkY])
                drawHands.draw_landmarks(frame, handDetails, fileHands.HAND_CONNECTIONS)
                gesture = gestureRecognizer.predict([handLandmarks])
                gestureIndex = np.argmax(gesture)
                gestureName = gestureNames[gestureIndex]
                print(gestureName)
                if(gestureName == "walkForward"):
                    print("Spot, take a step forward!")
                    #call function for Spot to walk forward
                if(gestureName == "walkBackward"):
                    print("Spot, take a step backward!")
                    #call function for Spot to walk backward
                if(gestureName == "walkLeft"):
                    print("Spot, take a step to the left!")
                    #call function for Spot to walk left
                if(gestureName == "walkRight"):
                    print("Spot, take a step to the right!")
                    #call function for Spot to walk right
                if(gestureName == "turnLeft"):
                    print("Spot, turn to the left!")
                    #call function for Spot to turn left
                if(gestureName == "turnRight"):
                    print("Spot, turn to the right!")
                    #call function for Spot to turn right
                if(gestureName == "tiltUp"):
                    print("Spot, tilt up!")
                    #call function for Spot to tilt up
                if(gestureName == "tiltDown"):
                    print("Spot, tilt down!")
                    #call function for Spot to tilt down
                #potential implementation of time.sleep()
        cv.imshow("Gesture", frame) 
        if (cv.waitKey(1) == ord("x")):
            break
    vid.release()
    cv.destroyAllWindows()

main()

# okay symbol = walkForward
# peace symbol = walkBackward (might need to tilt hand right a little)
# fist symbol = turnRight
# stop symbol = turnLeft
# thumbs up = tiltUp (make sure inner of thumb is facing camera)
# thumbs down = tiltDown
# call me symbol = walkLeft (make sure thumb is up and pinky is sideways to the left)
# reverse call me symbol = walkRight (this one is very finnicky, have to turn wrist, keep thumb up, and have pinky facing forward)
