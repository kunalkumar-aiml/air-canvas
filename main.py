import cv2
import time
import config
from hand_tracker import HandTracker
from drawing import Drawer
from logger import setup_logger


def main():

    setup_logger()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    tracker = HandTracker()
    drawer = Drawer()

    prev_time = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        drawer.initialize_canvas(frame)

        # Draw color buttons
        x_offset = 10
        color_positions = []

        for name, color in config.COLORS.items():

            x1 = x_offset
            x2 = x_offset + 100

            cv2.rectangle(
                frame,
                (x1, 10),
                (x2, config.UI_HEIGHT),
                color,
                -1
            )

            color_positions.append((x1, x2, color))

            x_offset += 110

        results = tracker.detect(frame)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                x, y = tracker.get_index_tip(frame, hand_landmarks)

                fingers = tracker.fingers_up(hand_landmarks)

                finger_count = sum(fingers)

                # DRAW MODE
                if finger_count == 1 and fingers[0] == 1:

                    drawer.draw(x, y)

                # COLOR SELECT MODE
                elif finger_count == 2:

                    drawer.reset_position()

                    if y < config.UI_HEIGHT:

                        for (x1, x2, color) in color_positions:

                            if x1 < x < x2:

                                drawer.set_color(color)

                # ERASE MODE
                elif finger_count >= 4:

                    drawer.erase(x, y)

                else:

                    drawer.reset_position()

                tracker.draw_landmarks(frame, hand_landmarks)

        output = drawer.get_output(frame)

        # FPS counter
        current_time = time.time()

        fps = 1 / (current_time - prev_time) if prev_time else 0

        prev_time = current_time

        cv2.putText(
            output,
            f"FPS: {int(fps)}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            "1 finger: Draw | 2 fingers: Select Color | Open hand: Erase",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Air Canvas", output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            drawer.save_canvas()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
