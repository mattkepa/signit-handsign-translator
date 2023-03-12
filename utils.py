import cv2
import copy
import itertools



# CONSTANTS
COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'green': (88, 209, 48),
    'red': (58, 69, 255),
    'purple': (150, 69, 149),
    'purple2': (242, 90, 191),
}
LM_CONN_COLOR = COLORS['green']
LM_POINT_COLOR = COLORS['red']
LM_CONN_THICKNESS = 2
LM_POINT_RADIUS = 5
BBOX_COLOR = COLORS['purple2']
BBOX_THICKNESS = 2
LABEL_COLOR = COLORS['purple']
LABEL_TEXT_COLOR = COLORS['white']



def calc_landmarks_pos(image, landmarks):
    """
    Calculates absolute position coordinates in pixels for detected hand's landmarks

    :param image: Mat - cv2 captured frame
    :param landmarks: list[list] - list of coordinates of hand landmarks returned from mediapipe detector
    """
    img_height = image.shape[0]
    img_width = image.shape[1]

    landmarks_pos = []

    for _, landmark in enumerate(landmarks.landmark):
        x = min(int(landmark.x * img_width), img_width - 1)
        y = min(int(landmark.y * img_height), img_height - 1)

        landmarks_pos.append([x, y])

    return landmarks_pos


def calc_bbox(landmarks):
    """
    Calculates x and y of top-left corner and width and height for bounding box of detected hand

    :param landmarks: list[list] - list of absolute coordinates pair of hand landmarks
    """
    x_positions = [point[0] for point in landmarks]
    y_positions = [point[1] for point in landmarks]
    min_x_pos = min(x_positions)
    max_x_pos = max(x_positions)
    min_y_pos = min(y_positions)
    max_y_pos = max(y_positions)
    width = abs(min_x_pos - max_x_pos)
    height = abs(min_y_pos - max_y_pos)

    return (min_x_pos, min_y_pos, width, height)



def preprocess_landmarks(landmarks):
    """
    Converts and normalize hand landmarks coordinates for machine learning model

    :param landmarks: list[list] - list of absolute coordinates pairs of hand landmarks
    """
    temp_landmarks = copy.deepcopy(landmarks)

    # Convert to relative coordinates
    # wrist landmark is at pos (0,0) and every other landmark is relative to it
    base_x, base_y = 0, 0
    for idx, point in enumerate(temp_landmarks):
        if idx == 0:
            base_x, base_y = point[0], point[1]

        temp_landmarks[idx][0] = temp_landmarks[idx][0] - base_x
        temp_landmarks[idx][1] = temp_landmarks[idx][1] - base_y

    # Convert to 1-dimensional list
    temp_landmarks = list(itertools.chain.from_iterable(temp_landmarks))

    # Normalization
    max_val = max(list(map(abs, temp_landmarks)))
    normalize = lambda n: n / max_val
    temp_landmarks = list(map(normalize, temp_landmarks))

    return temp_landmarks



def draw_landmarks(img, points):
    """
    Draws hand landmarks and its connections.
    Returns edited frame

    :param img: Mat - cv2 captured frame
    :param points: list[list] - list of absolute coordinates pairs of hand landmarks
    """
    if len(points) > 0:
        # CONNECTIONS
        #  PALM
        cv2.line(img, tuple(points[0]), tuple(points[1]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[0]), tuple(points[5]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[0]), tuple(points[17]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[5]), tuple(points[9]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[9]), tuple(points[13]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[13]), tuple(points[17]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        #  THUMB
        cv2.line(img, tuple(points[1]), tuple(points[2]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[2]), tuple(points[3]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[3]), tuple(points[4]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        #  INDEX FINGER
        cv2.line(img, tuple(points[5]), tuple(points[6]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[6]), tuple(points[7]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[7]), tuple(points[8]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        #  MIDDLE FINGER
        cv2.line(img, tuple(points[9]), tuple(points[10]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[10]), tuple(points[11]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[11]), tuple(points[12]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        #  RING FINGER
        cv2.line(img, tuple(points[13]), tuple(points[14]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[14]), tuple(points[15]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[15]), tuple(points[16]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        #  PINKY FINGER
        cv2.line(img, tuple(points[17]), tuple(points[18]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[18]), tuple(points[19]), LM_CONN_COLOR, LM_CONN_THICKNESS)
        cv2.line(img, tuple(points[19]), tuple(points[20]), LM_CONN_COLOR, LM_CONN_THICKNESS)

        # LANDMARK POINTS
        for landmark in points:
            cv2.circle(img, tuple(landmark), LM_POINT_RADIUS, LM_POINT_COLOR, -1)

    return img


def draw_bbox(img, bbox, corner_radius):
    """
    Draws bounding box of detected hand.
    Returns edited frame

    :param img: list - cv2 captured frame
    :param bbox: tuple - calculated x and y of top-left corner and width and height of bounding box
    :param corner_radius: int - radius of bottom corners
    """
    offset = 20
    top_left = (bbox[0] - offset, bbox[1] - offset)
    top_right = (bbox[0] + bbox[2] + offset, bbox[1] - offset)
    bottom_left = (bbox[0] - offset, bbox[1] + bbox[3] + offset)
    bottom_right = (bbox[0] + bbox[2] + offset, bbox[1] + bbox[3] + offset)

    cv2.line(img, top_left, top_right, BBOX_COLOR, BBOX_THICKNESS)
    cv2.line(img, top_left, (bottom_left[0], bottom_left[1] - corner_radius), BBOX_COLOR, BBOX_THICKNESS)
    cv2.line(img, (bottom_left[0] + corner_radius, bottom_left[1]), (bottom_right[0] - corner_radius, bottom_right[1]), BBOX_COLOR, BBOX_THICKNESS)
    cv2.line(img, top_right, (bottom_right[0], bottom_right[1] - corner_radius), BBOX_COLOR, BBOX_THICKNESS)

    cv2.ellipse(img, (bottom_left[0] + corner_radius, bottom_left[1] - corner_radius), (corner_radius, corner_radius), 90, 0, 90, BBOX_COLOR, BBOX_THICKNESS)
    cv2.ellipse(img, (bottom_right[0] - corner_radius, bottom_right[1] - corner_radius), (corner_radius, corner_radius), 0, 0, 90, BBOX_COLOR, BBOX_THICKNESS)

    return img


def draw_label(img, bbox, text):
    """
    Draws label for detected hand sign.
    Returns edited frame

    :param img: list - cv2 captured frame
    :param bbox: tuple - calculated x and y of top-left corner and width and height of hand bounding box
    :param text: str - label for detected hand sign
    """
    offset = 20
    height = 60
    width = bbox[2]
    start_x = bbox[0] - offset - BBOX_THICKNESS // 2
    start_y = bbox[1] - offset - height - BBOX_THICKNESS // 2
    end_x = bbox[0] + width + offset + BBOX_THICKNESS // 2
    end_y = bbox[1] - offset + BBOX_THICKNESS // 2

    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), LABEL_COLOR, -1)
    cv2.putText(img, text, (bbox[0], bbox[1] - 2*offset), cv2.FONT_HERSHEY_SIMPLEX, 1, LABEL_TEXT_COLOR, 2, cv2.LINE_AA)

    return img
