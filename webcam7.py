import cv2
import dlib
import numpy as np
import firebase_admin
import json
from firebase_admin import credentials
from firebase_admin import db
import win32api
import win32con
import telegram
from scipy.spatial import distance
import pyautogui


def calculate_EAR(eye): # 눈 거리 계산
   A = distance.euclidean(eye[1], eye[5])
   B = distance.euclidean(eye[2], eye[4])
   
   C = distance.euclidean(eye[0], eye[3])
   ear_aspect_ratio = (A+B)/(2.0*C)
   return ear_aspect_ratio
 

def shape_to_np(shape, dtype="int"):
   # initialize the list of (x, y)-coordinates
   coords = np.zeros((68, 2), dtype=dtype)
   # loop over the 68 facial landmarks and convert them
   # to a 2-tuple of (x, y)-coordinates
   for i in range(0, 68):
      coords[i] = (shape.part(i).x, shape.part(i).y)
   # return the list of (x, y)-coordinates
   return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask,points, 255)
    return mask

def contouring(thresh, mid, img):
        
    global map,EAR
    p=0
    p1=0
 
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        p=cx
        p1=cy
        c=map(p,0,640,0,1920)
        c1=map(p1,0,480,0,1080)
        c2 =int(c) # 매핑 값
        c3 = int(c1)# 매핑 값
        #print(p,p1)
     
        return cx,cy
        
    except:
        return 0,0
        pass
def map(x,input_min,input_max,output_min,output_max):
    return (x-input_min)*(output_max-output_min)/(input_max-input_min)+output_min #map()함수 정의



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)


#firebase 설정
cred = credentials.Certificate('')
firebase_admin.initialize_app(cred,{''})
mouse_x = 1920/2
mouse_y = 1080/2
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        global EAR

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        cx,cy = contouring(thresh[:,0:mid], mid, img)
      
        leftEye=[]
        rightEye=[]
        

        water=[]
        medicine=[]
        call=[]
        
        faces = detector(gray)
        face_landmarks = predictor(gray, rect)
        
        # 눈깜빡임 감지
        for n in range(36,42):# 오른쪽 눈 감지
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
               next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,255,0),1)
        
        
        
        leftEye = np.array(leftEye)
        leftEye = [leftEye[:,0].min(), #x좌표
                   leftEye[:,1].min(), #y좌표
                   leftEye[:,0].max()-leftEye[:,0].min(),
                   leftEye[:,1].max()-leftEye[:,1].min()]
        leftEyeCenter = [leftEye[0]+ leftEye[2]/2 -2, leftEye[1]+leftEye[3]/2-1]
        
        errX, errY = (leftEyeCenter[0]-cx,leftEyeCenter[1]-cy)

        
        if 2 < abs(errX)  < 20: 
                  win32api.SetCursorPos((int(mouse_x), int(mouse_y)))
                  #mouse_x -= errX*5
                  mouse_y -= errY*10
        if mouse_x < 0: mouse_x = 0
        if mouse_x <0: mouse_x = 0
        if mouse_x >1920: mouse_x = 1920
        if mouse_y <0: mouse_y = 0
        if mouse_y >1080: mouse_y = 1080

        #print(mouse_x,mouse_y)
        print(errX,errY)

        for n in range(42,48): # 왼쪽 눈 감지
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                    next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(img,(x,y),(x2,y2),(0,255,0),1)
        right_ear = calculate_EAR(rightEye)
        EAR = (right_ear+right_ear+0.1)/2
        EAR = round(EAR,2)
        
        print(EAR)

        if (EAR<0.16):
                print("눈깜빡임")
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, cx, cy, 0, 0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, cx, cy, 0, 0)
                
                ref = db.reference('firebase/water/')
                water = ref.get()

                ref1 = db.reference('firebase/medicine/')
                medicine =ref1.get()

                ref2 = db.reference('firebase/call/')
                call =ref2.get()

                if(water =="w"):
                        telgm_token = '5603269776:AAEpAA3fDfaZ09w4eTDOQlIEBt8ZptNcR7I'
                        bot = telegram.Bot(token = telgm_token)
                        bot.sendMessage(chat_id = '5634334517', text="물주세요")
                        print(water)
                        break

                if(medicine == "m"):
                        telgm_token = '5603269776:AAEpAA3fDfaZ09w4eTDOQlIEBt8ZptNcR7I'
                        bot = telegram.Bot(token = telgm_token)
                        print(medicine)
                        bot.sendMessage(chat_id = '5634334517', text="약주세요")
                        break

                if(call =="c"):
                        telgm_token = '5603269776:AAEpAA3fDfaZ09w4eTDOQlIEBt8ZptNcR7I'
                        bot = telegram.Bot(token = telgm_token)
                        bot.sendMessage(chat_id = '5634334517', text="의사를를 불러주세요!")
                        print(call)
                        break
                else:
                        print("아무것도 감지 x")
        
        print("Error:")
        print(errX,errY)
        print("상대좌표")
        print(mouse_x,mouse_y)
                        
                                
        
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
