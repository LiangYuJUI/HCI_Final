import cv2
import mediapipe as mp
import numpy as np
#import pyautogui
import json

with open('weight.json', 'r') as file:
    weight = json.load(file)

different_list = []

my_mp_drawing = mp.solutions.drawing_utils
my_mp_pose = mp.solutions.pose

image_mp_drawing = mp.solutions.drawing_utils
image_mp_pose = mp.solutions.pose

pose = my_mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 範例圖片的mp模組
image_pose = image_mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# cv2相機模組
cap = cv2.VideoCapture(0)

# cv2讀取範例影片
video_path = './example.mp4'
image_cap = cv2.VideoCapture(video_path)

# cv2讀取範例圖片
# image_path = './1.jpg'
# image_frame = cv2.imread(image_path)

# cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Output', 1000, 600)

# 獲取螢幕寬度和高度
#screen_width, screen_height = pyautogui.size()

#變數
landmark_dic_x= dict()
landmark_dic_y= dict()
landmark_img_x= dict()
landmark_img_y= dict()

pose_flag= dict()

#計算姿勢標準
def calculate(landmark_x, landmark_y, landmark_img_x, landmark_img_y):
    for key in ['leftarm', 'rightarm', 'leftthigh', 'rightthigh', 'leftforearm', 'rightforearm']:
        pose_flag[key] = True
    
    #手臂標準度
    leftarm= slopee(landmark_x[11], landmark_y[11], landmark_x[13], landmark_y[13])
    rightarm= slopee(landmark_x[12], landmark_y[12], landmark_x[14], landmark_y[14])
    img_leftarm= slopee(landmark_img_x[11], landmark_img_y[11], landmark_img_x[13], landmark_img_y[13])
    img_rightarm= slopee(landmark_img_x[12], landmark_img_y[12], landmark_img_x[14], landmark_img_y[14])
    #大腿標準度
    leftthigh= slopee(landmark_x[23], landmark_y[23], landmark_x[25], landmark_y[25])
    rightthigh= slopee(landmark_x[24], landmark_y[24], landmark_x[26], landmark_y[26])
    img_leftthigh= slopee(landmark_img_x[23], landmark_img_y[23], landmark_img_x[25], landmark_img_y[25])
    img_rightthigh= slopee(landmark_img_x[24], landmark_img_y[24], landmark_img_x[26], landmark_img_y[26])
    #小臂標準度
    leftforearm= slopee(landmark_x[13], landmark_y[13], landmark_x[15], landmark_y[15])
    rightforearm= slopee(landmark_x[14], landmark_y[14], landmark_x[16], landmark_y[16])
    img_leftforearm= slopee(landmark_img_x[13], landmark_img_y[13], landmark_img_x[15], landmark_img_y[15])
    img_rightforearm= slopee(landmark_img_x[14], landmark_img_y[14], landmark_img_x[16], landmark_img_y[16])
    
    # 誤差值
    leftarm_error = error(leftarm, img_leftarm)
    rightarm_error = error(rightarm, img_rightarm)
    leftthigh_error = error(leftthigh, img_leftthigh)
    rightthigh_error = error(rightthigh, img_rightthigh)
    leftforearm_error = error(leftforearm, img_leftforearm)
    rightforearm_error = error(rightforearm, img_rightforearm)

    difference = (leftarm_error * weight["leftarm"]) + (rightarm_error * weight["rightarm"]) + (leftthigh_error * weight["leftthigh"]) + (rightthigh_error * weight["rightthigh"]) + (leftforearm_error * weight["leftforearm"]) + (rightforearm_error * weight["rightforearm"])
    #print(type(difference))
    if difference < 50 and difference > 0:
        different_list.append(difference)
    else:
        different_list.append(0)

    # 計算精準是否 與 誤差值
    if(camera_exceed(landmark_img_x[11], landmark_img_y[11], landmark_img_x[13], landmark_img_y[13])):
        if(not (approx(leftarm, img_leftarm) and camera_exceed(landmark_x[11], landmark_y[11], landmark_x[13], landmark_y[13]))):
            pose_flag['leftarm']= False
    
    if(camera_exceed(landmark_img_x[12], landmark_img_y[12], landmark_img_x[14], landmark_img_y[14])):
        if(not (approx(rightarm, img_rightarm) and camera_exceed(landmark_x[12], landmark_y[12], landmark_x[14], landmark_y[14]))):
            pose_flag['rightarm']= False
    
    if(camera_exceed(landmark_img_x[13], landmark_img_y[13], landmark_img_x[15], landmark_img_y[15])):
        if(not (approx(leftforearm, img_leftforearm) and camera_exceed(landmark_x[13], landmark_y[13], landmark_x[15], landmark_y[15]))):
            pose_flag['leftforearm']= False
    
    if(camera_exceed(landmark_img_x[14], landmark_img_y[14], landmark_img_x[16], landmark_img_y[16])):
        if(not (approx(rightforearm, img_rightforearm) and camera_exceed(landmark_x[14], landmark_y[14], landmark_x[16], landmark_y[16]))):
            pose_flag['rightforearm']= False
    
    if(camera_exceed(landmark_img_x[23], landmark_img_y[23], landmark_img_x[25], landmark_img_y[25])):
        if(not (approx(leftthigh, img_leftthigh) and camera_exceed(landmark_x[23], landmark_y[23], landmark_x[25], landmark_y[25]))):
            pose_flag['leftthigh']= False
    
    if(camera_exceed(landmark_img_x[24], landmark_img_y[24], landmark_img_x[26], landmark_img_y[26])):
        if(not (approx(rightthigh, img_rightthigh) and camera_exceed(landmark_x[24], landmark_y[24], landmark_x[26], landmark_y[26]))):
            pose_flag['rightthigh']= False
 
#計算誤差值
def error(x, y):
    return abs(x-y)

#計算接近值
def approx(x, y):
    if(abs(x-y)> 3):
        return False
    return True
        
#節點是否超出鏡頭
def camera_exceed(x1, y1, x2, y2):
    if x1> 1 or x1< 0:
        return False
    elif y1> 1 or y1< 0:
        return False
    elif x2> 1 or x2< 0:
        return False
    elif y2> 1 or y2< 0:
        return False
    return True

#計算斜率
def slopee(x1, y1, x2, y2):
    x = (y2 - y1) / (x2 - x1)
    return x

#Main function
while cap.isOpened() and image_cap.isOpened():
    ret, frame = cap.read()
    image_ret, image_frame = image_cap.read()

    try:
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        results = pose.process(RGB)

        image_RGB = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
        image_results = image_pose.process(image_RGB)

        #print(type(results.pose_landmarks))
        # 在主迴圈中的適當位置取得landmark
        
        for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
            # 取得X、Y、Z坐標
            landmark_dic_x[landmark_id]= landmark.x
            landmark_dic_y[landmark_id]= landmark.y
            # 打印或使用這些坐標值
            #print(f"Landmark {landmark_id}: X={landmark_x}, Y={landmark_y}, Z={landmark_z}")
        for landmark_id, landmark in enumerate(image_results.pose_landmarks.landmark):
            landmark_img_x[landmark_id]= landmark.x
            landmark_img_y[landmark_id]= landmark.y
        
        calculate(landmark_dic_x, landmark_dic_y, landmark_img_x, landmark_img_y)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmark.x = 1 - landmark.x
        
        my_mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, my_mp_pose.POSE_CONNECTIONS)
        image_mp_drawing.draw_landmarks(
            image_frame, image_results.pose_landmarks, image_mp_pose.POSE_CONNECTIONS)
        
        # 在背景上顯示文字 or 動作評判顯示
        if(pose_flag['leftarm']):
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.RIGHT_SHOULDER,my_mp_pose.PoseLandmark.RIGHT_ELBOW)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6))
        else:
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.RIGHT_SHOULDER,my_mp_pose.PoseLandmark.RIGHT_ELBOW)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6))
        if(pose_flag['rightarm']):
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.LEFT_SHOULDER,my_mp_pose.PoseLandmark.LEFT_ELBOW)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6))
        else:
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.LEFT_SHOULDER,my_mp_pose.PoseLandmark.LEFT_ELBOW)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6))
        if(pose_flag['leftthigh']):
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.RIGHT_HIP,my_mp_pose.PoseLandmark.RIGHT_KNEE)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6))
        else:
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.RIGHT_HIP,my_mp_pose.PoseLandmark.RIGHT_KNEE)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6))
        if(pose_flag['rightthigh']):
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.LEFT_HIP,my_mp_pose.PoseLandmark.LEFT_KNEE)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6))
        else:
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.LEFT_HIP,my_mp_pose.PoseLandmark.LEFT_KNEE)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6))
        if(pose_flag['leftforearm']):
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.RIGHT_ELBOW,my_mp_pose.PoseLandmark.RIGHT_WRIST)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6))
        else:
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.RIGHT_ELBOW,my_mp_pose.PoseLandmark.RIGHT_WRIST)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6))
        if(pose_flag['rightforearm']):
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.LEFT_ELBOW,my_mp_pose.PoseLandmark.LEFT_WRIST)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=6))
        else:
            my_mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, [(my_mp_pose.PoseLandmark.RIGHT_ELBOW,my_mp_pose.PoseLandmark.RIGHT_WRIST)], connection_drawing_spec=my_mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6))
       
        if frame.shape[0] != image_frame.shape[0]:
            image_frame = cv2.resize(image_frame, (frame.shape[1], frame.shape[0]))
        
        combined_image = np.hstack((frame, image_frame))  # 垂直堆疊 frame 和 image_frame
        # 創建一塊黑畫布
        black_canvas = np.zeros((combined_image.shape[0]*2, combined_image.shape[1]*2, 3), dtype=np.uint8)
        # 將 combined_image 貼在畫布上
        black_canvas[0:0+combined_image.shape[0], 0:0+combined_image.shape[1]] = combined_image
        
        
        #若全對，則動作通過
        tmp = all(val for key, val in pose_flag.items())
        if(tmp):
            cv2.putText(black_canvas, 'POSTURE PASS ! ! !', (900, 800), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(black_canvas, 'POSTURE UNPASS ><', (900, 800), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Output', black_canvas)
        #cv2.imshow('Output', np.hstack((frame, image_frame)))
    except:
        break
        
    key = cv2.waitKey(1)
    if key == 27:  # Check for the Esc key
        break

cap.release()
cv2.destroyAllWindows()
print(sum(different_list)/len(different_list))