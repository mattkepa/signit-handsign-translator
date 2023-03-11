import cv2


def main():
    # Camera configuration
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Listen for ESC key press to exit
        key = cv2.waitKey(10)
        if key == 27:
            break

        # Display output
        cv2.imshow('SignIT', frame)


    # Close window and release camera
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()