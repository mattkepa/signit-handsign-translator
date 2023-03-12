import cv2
import mediapipe as mp
import csv
import copy
from model.handsign_classifier import HandSignClassifier
from utils import *


def main():
    # Camera configuration
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # Hand detection model configuration
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    # Hand sign classifier configuration
    classifier = HandSignClassifier()
    with open('model/handsign_labels.csv', 'r', encoding='utf-8-sig') as f:  # read labels for classifier
        handsign_labels = csv.reader(f)
        handsign_labels = [row[0] for row in handsign_labels]


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Listen for ESC key press to exit
        key = cv2.waitKey(10)
        if key == 27:
            break

        img = cv2.flip(frame, 1) # flip image horizontal for hand detection model

        output = copy.deepcopy(img) # create copy of img to display as output in BGR color mode

        # Detect hand
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert img mode BGR to RGB
        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = calc_landmarks_pos(img, hand_landmarks) # get absolute position hand landmarks coordinates
                bbox = calc_bbox(landmarks) # calculate bounding box for hand

                # Convert detected hand landmarks to relative and normalized coordinates
                processed_landmarks = preprocess_landmarks(landmarks)

                # Hand sign landmarks data classification
                hand_sign_idx = classifier(processed_landmarks) # returns index of classified sign

                # Draw information to ouput image
                output = draw_bbox(output, bbox, 20)
                if hand_sign_idx is not None:
                    output = draw_label(output, bbox, handsign_labels[hand_sign_idx])
                output = draw_landmarks(output, landmarks)

        # Display output
        cv2.imshow('SignIT', output)


    # Close window and release camera
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()