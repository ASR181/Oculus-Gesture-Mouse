import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # initialize mediapipe hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # landmarks for fingertips
        self.tip_ids =[4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True, color=(255, 229, 0)):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # apply custom skeleton color based on the current theme
                    landmark_spec = self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3)
                    connection_spec = self.mp_draw.DrawingSpec(color=color, thickness=2)
                    
                    self.mp_draw.draw_landmarks(
                        img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                        landmark_spec, connection_spec
                    )
        return img

    def find_position(self, img, hand_no=0, draw=True):
        x_list = []
        y_list = []
        bbox = []
        self.lm_list =[]
        
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([id, cx, cy])
                
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = xmin, ymin, xmax, ymax

        return self.lm_list, bbox

    def fingers_up(self):
        fingers =[]
        
        if not hasattr(self, 'lm_list') or len(self.lm_list) == 0:
            return [0, 0, 0, 0, 0]

        # thumb check
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # check the other 4 fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def find_distance(self, p1, p2, img, draw=True, r=10, t=3, color=(0, 255, 0)):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), color, t)
            cv2.circle(img, (cx, cy), r, color, cv2.FILLED)
            
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img,[x1, y1, x2, y2, cx, cy]