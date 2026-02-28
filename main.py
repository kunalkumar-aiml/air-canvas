import cv2
from hand_tracker import HandTracker
from drawing import Drawer


def main():
    cap = cv2.VideoCapture(0)

    tracker = HandTracker()
    drawer = Drawer()

    colors = {
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255)
    }

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        drawer.initialize_canvas(frame)

        # Draw color buttons
        cv2.rectangle(frame, (10, 10), (110, 60), colors["blue"], -1)
        cv2.rectangle(frame, (120, 10), (220, 60), colors["green"], -1)
        cv2.rectangle(frame, (230, 10), (330, 60), colors["red"], -1)
        cv2.rectangle(frame, (340, 10), (440, 60), colors["yellow"], -1)

        results = tracker.detect(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x, y = tracker.get_index_tip(frame, hand_landmarks)
                fingers = tracker.fingers_up(hand_landmarks)

                # Only index finger up → Draw
                if fingers[0] == 1 and sum(fingers) == 1:
                    drawer.draw(x, y)

                # Two fingers up → Erase
                elif fingers[0] == 1 and fingers[1] == 1:
                    drawer.clear_canvas()

                else:
                    drawer.reset_position()

                # Color selection
                if y < 60:
                    if 10 < x < 110:
                        drawer.set_color(colors["blue"])
                    elif 120 < x < 220:
                        drawer.set_color(colors["green"])
                    elif 230 < x < 330:
                        drawer.set_color(colors["red"])
                    elif 340 < x < 440:
                        drawer.set_color(colors["yellow"])

                tracker.draw_landmarks(frame, hand_landmarks)

        output = drawer.get_output(frame)

        cv2.putText(output, "Q - Quit | S - Save",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

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
