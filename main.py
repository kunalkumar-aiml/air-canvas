import cv2
import config
from hand_tracker import HandTracker
from drawing import Drawer


def main():

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    tracker = HandTracker()
    drawer = Drawer()

    tool = "draw"

    while True:

        success,frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame,1)

        drawer.initialize_canvas(frame)

        # TOOL BUTTONS
        cv2.rectangle(frame,(10,10),(120,70),(50,50,50),-1)
        cv2.putText(frame,"COLOR",(20,50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.rectangle(frame,(130,10),(240,70),(50,50,50),-1)
        cv2.putText(frame,"BRUSH",(140,50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.rectangle(frame,(260,10),(370,70),(50,50,50),-1)
        cv2.putText(frame,"SHAPE",(270,50),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        results = tracker.detect(frame)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                x,y = tracker.get_index_tip(frame,hand)

                fingers = tracker.fingers_up(hand)

                count = sum(fingers)

                if count == 1:

                    if tool == "draw":
                        drawer.draw(x,y)

                    elif tool == "erase":
                        drawer.erase(x,y)

                elif count == 2:

                    drawer.reset()

                    if y < config.UI_HEIGHT:

                        if 10 < x < 120:
                            tool = "color"

                        elif 130 < x < 240:
                            tool = "brush"

                        elif 260 < x < 370:
                            tool = "shape"

                elif count >= 4:

                    drawer.erase(x,y)

        output = drawer.output(frame)

        cv2.imshow("AirCanvas",output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
