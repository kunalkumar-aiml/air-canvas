import cv2
import time
import config
from hand_tracker import HandTracker
from drawing import Drawer


def main():

    cap=cv2.VideoCapture(config.CAMERA_INDEX)

    tracker=HandTracker()
    drawer=Drawer()

    prev_time=0

    while True:

        success,frame=cap.read()

        if not success:
            break

        frame=cv2.flip(frame,1)

        drawer.initialize(frame)

        # draw color palette
        x_offset=10

        color_positions=[]

        for color in config.COLORS:

            x1=x_offset
            x2=x_offset+60

            cv2.rectangle(frame,(x1,10),(x2,70),color,-1)

            color_positions.append((x1,x2,color))

            x_offset+=70

        results=tracker.detect(frame)

        if results.multi_hand_landmarks:

            for hand in results.multi_hand_landmarks:

                x,y=tracker.get_index_tip(frame,hand)

                fingers=tracker.fingers_up(hand)

                count=sum(fingers)

                # DRAW
                if count==1 and fingers[0]==1:

                    drawer.draw(x,y)

                # COLOR SELECT
                elif count==2:

                    drawer.reset()

                    if y<config.UI_HEIGHT:

                        for (x1,x2,color) in color_positions:

                            if x1<x<x2:

                                drawer.set_color(color)

                # ERASE
                elif count>=4:

                    drawer.erase(x,y)

                else:

                    drawer.reset()

                tracker.draw_landmarks(frame,hand)

        output=drawer.output(frame)

        # FPS
        current=time.time()

        fps=1/(current-prev_time) if prev_time else 0

        prev_time=current

        cv2.putText(
            output,
            f"FPS:{int(fps)}",
            (10,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            output,
            "1 finger draw | 2 fingers select | open hand erase",
            (10,130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        cv2.imshow("AirCanvas",output)

        key=cv2.waitKey(1)&0xFF

        if key==ord("q"):
            break

        elif key==ord("s"):
            drawer.save()

        elif key==ord("c"):
            drawer.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
