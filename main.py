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

        # UI Buttons
        x_offset = 10

        for name, color in config.COLORS.items():

            cv2.rectangle(
                frame,
                (x_offset, 10),
                (x_offset + 100, config.UI_HEIGHT),
                color,
                -1
            )

            x_offset += 110

        # Erase button
        cv2.rectangle(
            frame,
            (x_offset, 10),
            (x_offset + 100, config.UI_HEIGHT),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "Erase",
            (x_offset + 10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        results = tracker.detect(frame)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                x, y = tracker.get_index_tip(frame, hand_landmarks)

                fingers = tracker.fingers_up(hand_landmarks)

                # Draw mode
                if fingers[0] == 1 and sum(fingers) == 1:

                    drawer.draw(x, y)

                # Selection mode
                elif fingers[0] == 1 and fingers[1] == 1:

                    drawer.reset_position()

                    if y < config.UI_HEIGHT:

                        x_offset = 10

                        for name, color in config.COLORS.items():

                            if x_offset < x < x_offset + 100:
                                drawer.set_color(color)

                            x_offset += 110

                        if x_offset < x < x_offset + 100:
                            drawer.enable_eraser()

                else:
                    drawer.reset_position()

                tracker.draw_landmarks(frame, hand_landmarks)

        output = drawer.get_output(frame)

        # FPS Counter
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
            "Q - Quit | S - Save",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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
