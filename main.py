import cv2
from hand_tracker import HandTracker
from drawing import Drawer


def main():
    cap = cv2.VideoCapture(0)

    tracker = HandTracker()
    drawer = Drawer()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        drawer.initialize_canvas(frame)

        results = tracker.detect(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x, y = tracker.get_index_tip(frame, hand_landmarks)
                drawer.draw(frame, x, y)
                tracker.draw_landmarks(frame, hand_landmarks)
        else:
            drawer.reset_position()

        output = drawer.get_output(frame)
        cv2.imshow("Air Canvas", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
