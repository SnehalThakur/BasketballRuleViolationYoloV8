import cv2
import numpy as np
import time
from ultralytics import YOLO


class DoubleDribbleDetector:
    def __init__(self):
        self.pose_model = YOLO("yolov8s-pose.pt")
        self.ball_model = YOLO("basketballModel.pt")
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture(r'data/basketballViolation-DD.mp4')
        self.body_index = {"left_wrist": 10, "right_wrist": 9}
        self.hold_start_time = None
        self.is_holding = False
        self.was_holding = False
        self.hold_duration = 0.85
        self.hold_threshold = 300
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        self.dribble_threshold = 18
        self.double_dribble_time = None
        self.frame_width = int(self.cap.get(3))

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                pose_annotated_frame, ball_detected = self.process_frame(frame)
                self.check_double_dribble()
                if self.double_dribble_time and time.time() - self.double_dribble_time <= 3:
                    red_tint = np.full_like(pose_annotated_frame, (0, 0, 255), dtype=np.uint8)
                    pose_annotated_frame = cv2.addWeighted(pose_annotated_frame, 0.7, red_tint, 0.3, 0)
                    cv2.putText(pose_annotated_frame, "Double dribble detected!", (self.frame_width - 600, 150,), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA,)

                cv2.imshow("AI Basketball Referee - Double dribble detection ", pose_annotated_frame)
                # return pose_annotated_frame
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        pose_annotated_frame = pose_results[0].plot()
        rounded_results = pose_results[0].keypoints.cpu().numpy()

        try:
            left_wrist = rounded_results[0].data[0][self.body_index["left_wrist"]]
            right_wrist = rounded_results[0].data[0][self.body_index["right_wrist"]]
        except:
            print("No human detected.")
            return pose_annotated_frame, False

        ball_results_list = self.ball_model(frame, verbose=False, conf=0.65)
        ball_detected = False

        for ball_results in ball_results_list:
            for bbox in ball_results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                ball_x_center = (x1 + x2) / 2
                ball_y_center = (y1 + y2) / 2
                self.update_dribble_count(ball_x_center, ball_y_center)
                self.prev_x_center = ball_x_center
                self.prev_y_center = ball_y_center
                ball_detected = True
                left_distance = np.hypot(ball_x_center.cpu() - left_wrist[0], ball_y_center.cpu() - left_wrist[1])
                right_distance = np.hypot(ball_x_center.cpu() - right_wrist[0], ball_y_center.cpu() - right_wrist[1])
                self.check_holding(left_distance, right_distance)
                cv2.rectangle(
                    pose_annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    pose_annotated_frame,
                    f"Ball: ({ball_x_center:.2f}, {ball_y_center:.2f})",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(pose_annotated_frame, f"Left Wrist: ({left_wrist[0]:.2f}, {left_wrist[1]:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA,)
                cv2.putText(pose_annotated_frame, f"Right Wrist: ({right_wrist[0]:.2f}, {right_wrist[1]:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA,)
                # cv2.putText(pose_annotated_frame, f"Differentials: ({min(left_distance, right_distance):.2f})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA,)
                # cv2.putText(pose_annotated_frame, f"Holding: {'Yes' if self.is_holding else 'No'}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0), 2, cv2.LINE_AA,)
                #
                # cv2.putText(
                #     pose_annotated_frame,
                #     f"Dribble count: {self.dribble_count}",
                #     (10, 120),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (0, 0, 0),
                #     2,
                #     cv2.LINE_AA,
                # )
                if self.is_holding:
                    blue_tint = np.full_like(
                        pose_annotated_frame, (255, 0, 0), dtype=np.uint8
                    )
                    pose_annotated_frame = cv2.addWeighted(
                        pose_annotated_frame, 0.7, blue_tint, 0.3, 0
                    )

        if not ball_detected:
            self.hold_start_time = None
            self.is_holding = False

        return pose_annotated_frame, ball_detected

    def check_holding(self, left_distance, right_distance):
        if min(left_distance, right_distance) < self.hold_threshold:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            elif time.time() - self.hold_start_time > self.hold_duration:
                self.is_holding = True
                self.was_holding = True
                self.dribble_count = 0
        else:
            self.hold_start_time = None
            self.is_holding = False

    def update_dribble_count(self, x_center, y_center):
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center
            if (
                    self.prev_delta_y is not None
                    and delta_y < 0
                    and self.prev_delta_y > self.dribble_threshold
            ):
                self.dribble_count += 1

            self.prev_delta_y = delta_y

    def check_double_dribble(self):
        if self.was_holding and self.dribble_count > 0:
            self.double_dribble_time = time.time()
            self.was_holding = False
            self.dribble_count = 0
            print("Double dribble detected!")


# Create a DoubleDribbleDetector instance and start it.
if __name__ == "__main__":
    detector = DoubleDribbleDetector()
    detector.run()
