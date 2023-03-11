import cv2
import mediapipe as mp
import copy
import csv
from utils import *



def main():
    # Camera configuration
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # Hand detection model configuration
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    # Set mode of logging data
    # 0 - normal mode
    # 1 - mode for logging data
    mode = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Listen for key press to exit or change mode
        key = cv2.waitKey(10)
        if key == 27:
            break
        mode, letter_id = set_mode(key, mode)


        img = cv2.flip(frame, 1) # flip image horizontal for hand detection model

        output = copy.deepcopy(img)

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

                # Write data to dataset csv file
                write_data(mode, letter_id, processed_landmarks)

                # Draw information to ouput image
                output = draw_bbox(output, bbox, 20)
                output = draw_landmarks(output, landmarks)

        # Display ouput image
        cv2.imshow('SignIT - Data Collector', output)


    # Close window and release camera
    cap.release()
    cv2.destroyAllWindows()




def set_mode(key, mode):
    """
    Sets mode for storing data

    :param key: int - value of pressed key
    :param mode: int - mode of logging data
    """
    letter_id = -1
    if 48 <= key <= 49: # key: 0 or 1
        mode = key - 48
    if 97 <= key <= 123: # key: a-z
        # each letter [a-z] has a corresponding id [0-25]
        letter_id = key - 97

    return mode, letter_id


def write_data(mode, letter_id, data):
    """
    Writes data to dataset csv file

    :param mode: int - mode of logging data
    :param letter_id: int - id of letter to store data to it
    :param data: list - processed and normalized hand landmarks
    """
    if mode == 0:
        pass
    if mode == 1 and (0 <= letter_id <= 25):
        csv_path = 'model/handsign_data.csv'
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([letter_id, *data])



if __name__ == '__main__':
    main()