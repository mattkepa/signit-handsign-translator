import cv2



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
