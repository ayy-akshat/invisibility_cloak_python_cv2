import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")

outputfile = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)

print("0")

time.sleep(2)

print("1")

bg = 0

for i in range(60):
    ret, bg = cap.read()

print("background captured")

bg = np.flip(bg, axis=1)

i = 0

while cap.isOpened():
    print("while")
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,0,0])
    upper_black = np.array([360,255,50])
    mask_1 = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask_2 = cv2.bitwise_not(mask_1)

    res_1 = cv2.bitwise_and(img, img, mask=mask_2)
    res_2 = cv2.bitwise_and(bg, bg, mask=mask_1)

    final_output = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    # outputfile.write(final_output)
    
    cv2.imwrite("img" + str(i) + ".jpg", final_output)

    cv2.waitKey(1)

    i += 1
    if i >= 120:
        break

print("done")

cap.release()
cv2.destroyAllWindows()
