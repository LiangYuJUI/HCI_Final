import cv2
import mediapipe as mp
import numpy as np

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

# cv2讀取範例圖片
image_path = './1.jpg'
image_frame = cv2.imread(image_path)
print(image_frame)
# cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Output', 1000, 600)

#計算姿勢標準
def calculate(landmark_x, landmark_y, landmark_z):
    leftarm= slopee(landmark_x[11], landmark_y[11], landmark_x[13], landmark_y[13])
    rightarm= slopee(landmark_x[12], landmark_y[12], landmark_x[14], landmark_y[14])
    
    
    if(leftarm< -1.15 and camera_exceed(landmark_x[11], landmark_y[11], landmark_x[13], landmark_y[13])):
        print('左臂動作成功')
    
    if(rightarm> 1.15 and camera_exceed(landmark_x[12], landmark_y[12], landmark_x[14], landmark_y[14])):
        print('右臂動作成功')
        
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
while cap.isOpened():
    ret, frame = cap.read()

    #try:
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    results = pose.process(RGB)

    #print(type(results.pose_landmarks))
    # 在主迴圈中的適當位置取得landmark
    landmark_dic_x= dict()
    landmark_dic_y= dict()
    landmark_dic_z= dict()
    
    for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
        # 取得X、Y、Z坐標
        landmark_dic_x[landmark_id]= landmark.x
        landmark_dic_y[landmark_id]= landmark.y
        landmark_dic_z[landmark_id]= landmark.z
        
        # 打印或使用這些坐標值
        #print(f"Landmark {landmark_id}: X={landmark_x}, Y={landmark_y}, Z={landmark_z}")
    
    calculate(landmark_dic_x, landmark_dic_y, landmark_dic_z)
    
    my_mp_drawing.draw_landmarks(
        frame, results.pose_landmarks, my_mp_pose.POSE_CONNECTIONS)
    
    image_RGB = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    image_results = image_pose.process(image_RGB)
    image_mp_drawing.draw_landmarks(
        image_frame, image_results.pose_landmarks, image_mp_pose.POSE_CONNECTIONS)
    
    #cv2.imshow('Output', frame)
    #cv2.imshow('Output', image_frame)

    if frame.shape[0] != image_frame.shape[0]:
        image_frame = cv2.resize(image_frame, (frame.shape[1], frame.shape[0]))
        
    cv2.imshow('Output', np.hstack((frame, image_frame)))
    #except:
    #   break
    
    key = cv2.waitKey(1)
    if key == 27:  # Check for the Esc key
        break

cap.release()
cv2.destroyAllWindows()
