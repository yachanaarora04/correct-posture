import cv2
import time
import math as m
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
pTime = 0
cTime = 0
while True:
    success, image = cap.read()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    h, w = image.shape[:2]
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    def findDistance(x1, y1, x2, y2):
        dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist

    offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
    text = str(int(offset))
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (w - 150, 30)

    thickness=2
    fontScale=0.9

    cv2.putText(image, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)

    def findAngle(x1, y1, x2, y2):
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        degree = int(180 / m.pi) * theta
        return degree

    neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
    cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

    angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

    def sendWarning():
        pass

    good_frames = 0
    bad_frames = 0
    if neck_inclination < 40 and torso_inclination < 10:
        bad_frames = 0
        good_frames += 1

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 2)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 2)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 2)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 2)
    else:
        good_frames = 0
        bad_frames += 1

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 2)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 2)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 2)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 2)

    good_time = (1 / fps) * good_frames
    bad_time = (1 / fps) * bad_frames

    if good_time > 0:
        time_string_good = 'Good Posture Time : ' + str(round(good_time, 2)) + 's'
        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
    else:
        time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 2)) + 's'
        cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

    if bad_time > 180:
        sendWarning()

    cv2.imshow("image", image)
    cv2.waitKey(1)

