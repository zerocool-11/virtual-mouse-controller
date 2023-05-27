import pyautogui
import copy
import itertools
import cv2 as cv
import mediapipe as mp
import numpy as np


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

def draw_landmarks(image, landmark_point):
    
    if len(landmark_point) > 0:
        
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0,255,0), 2)

        
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0,255,0), 2)

        
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0,255,0), 2)

        
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0,255,0), 2)

        
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0,255,0), 2)

        
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0,255,0), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0,255,0), 2)

    
    for index, landmark in enumerate(landmark_point):
        if index % 4 != 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (0,0,255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 0, 0), 1)
        else:
            cv.circle(image, (landmark[0], landmark[1]), 8, (0,0,255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 0, 0), 1)
        

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list



def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


def draw_info_text(image, handedness, hand_sign_text):
    # cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
    #              (0, 0, 0), -1)

    info_text = handedness.classification[0].label
    if hand_sign_text != "":
        info_text = 'hand :' + info_text
    cv.putText(image, info_text, (25,45),
               cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv.LINE_AA)

    # if finger_gesture_text != "":
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)

    return image


def draw_word(image,word):
    word = "".join(word)

    cv.putText(image, "Search :"+word, (25,130),
               cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv.LINE_AA)

    return image
    

wscr,hscr=pyautogui.size()
riox,rioy=150,150
cwidth=940
cheight=535
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 5
count=0
ind=0
thb=0
mid=0
dist=0.0
dragndrop=False
finger_index="down"
finger_middle="down"
finger_thumb="down"

cam=cv.VideoCapture(1)
cam.set(cv.CAP_PROP_FRAME_WIDTH, cwidth)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, cheight)
mp_hands=mp.solutions.hands
hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

i=0
while True:
    res,frame=cam.read()

    if not res:
        break
    frame=cv.flip(frame,1)
    debug_image = copy.deepcopy(frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results=hands.process(frame)
    frame.flags.writeable = True
    image_height, image_width, _ = frame.shape
    if results.multi_hand_landmarks is not None:
        #for hand_landmarks in results.left_hand_landmarks:
        for hand_landmarks,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                wrist = np.array(landmark_list[0])
                ring=np.array(landmark_list[16])
                ring_s=np.array(landmark_list[13])
                pinky=np.array(landmark_list[20])
                pinky_s=np.array(landmark_list[17])
                index = np.array(landmark_list[8])
                index_ip=np.array(landmark_list[6])
                index_s=np.array(landmark_list[5])
                middle=np.array(landmark_list[12])
                middle_ip=np.array(landmark_list[10])
                middle_s=np.array(landmark_list[9])
                thumb=np.array(landmark_list[4])
                thumb_ip=np.array(landmark_list[3])
                thumb_s=np.array(landmark_list[1])
                sum_tip= np.sum(np.square(middle- index))
                dist=np.sqrt(sum_tip)
                sum_thb= np.sum(np.square(index-thumb))
                dist_thb=np.sqrt(sum_tip)
                length = np.sqrt(sum_thb)
                #cv.putText(debug_image, "thb distance :" + str(length), (10, 120), cv.FONT_HERSHEY_SIMPLEX,1.0, (255, 255, 255), 1, cv.LINE_AA)
                print("distance b/w thumb and index ",dist_thb)
                if index[1]<index_ip[1]:  # checks if finger is up 
                        finger_index="up"
                        ind=1
                        print("index up")
                if index[1]>index_ip[1]:
                        finger_index="down"
                        ind=0
                        print("index down")

                
                if middle[1]<middle_ip[1]: # checks if finger is up 
                        finger_middle="up"
                        mid=1
                        print("middle up")
                if middle[1]>middle_ip[1]:
                        finger_middle="down"
                        mid=0
                        print("middle down")

                
 
                if thumb[0]>thumb_ip[0]: # checks if thumb is up 
                        finger_thumb="up"
                        thb=1
                        print("thumbs up")
                if thumb[0]<thumb_ip[0]:
                        finger_thumb="down"
                        thb=0
                        print("thumb down")

                #cv.rectangle(debug_image, (landmark_list[8][0]-25,landmark_list[8][1]-15),(landmark_list[8][0]+25,landmark_list[8][1]+15), (255,0,0), 2)
                cv.rectangle(debug_image, (riox,rioy),(cwidth-riox,cheight-rioy), (255,255,0), 2) # a particular area which we can use for cursor movement
                if ind==1 and (mid==0 or dist>40):
                        x3 = np.interp(landmark_list[8][0], (riox, cwidth - riox), (0, wscr))
                        y3 = np.interp(landmark_list[8][1], (rioy, cheight - rioy), (0, hscr))
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening
                        if dragndrop:
                                pyautogui.dragTo(clocX, clocY, button='left')   # for drag nd drop 
                        else:
                                pyautogui.moveTo(clocX, clocY)
                        plocX, plocY = clocX, clocY
                        count=0     
                        # i+=1
                if ind==1 and mid==1 and dist <= 40.0:
                        if count==0:
                                pyautogui.click(button='left')  # single click
                                print("clicked")
                                count+=1
                
                if length<=30.0:
                        #pyautogui.mouseDown(button='left')
                        dragndrop=True
                        cv.putText(debug_image, "thb distance :" + str(length), (10, 120), cv.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 1, cv.LINE_AA)
                else:
                        dragndrop=False
                        cv.putText(debug_image, "thb distance :" + str(length), (10, 120), cv.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), 1, cv.LINE_AA)

        
        

                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                        debug_image,
                        handedness,
                        "None")
    if ind:
        landmark_list[5]
        if dist <= 40.0:
                
                cv.putText(debug_image, "distance :" + str(dist), (10, 90), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv.LINE_AA)
        else:
                cv.putText(debug_image, "distance :" + str(dist), (10, 90), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 2, cv.LINE_AA)
        
        cv.putText(debug_image, "index:" + str(finger_index), (10, 150), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 2, cv.LINE_AA)
        cv.putText(debug_image, "middle:" + str(finger_middle), (200, 150), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 2, cv.LINE_AA)
        cv.putText(debug_image, "thumb:" + str(finger_thumb), (450, 150), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 2, cv.LINE_AA)
        
    cv.imshow('Hand shape recognition', debug_image)
    key = cv.waitKey(10)
    if key == 27:  
        break

cv.destroyAllWindows()
cam.release()


    


# In[ ]:





# In[20]:


cv.destroyAllWindows()
cam.release()


# In[53]:





# In[ ]:




