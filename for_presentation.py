import cv2
import mediapipe as mp
import numpy as np
import time 
import serial
from tensorflow.keras.models import load_model
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound


py_serial_main = serial.Serial(
    # Window
    port='COM15',
    
    # 보드 레이트 (통신 속도)
    baudrate=115200,
)

py_serial_sub = serial.Serial(
    # Window
    port='COM16',
    
    # 보드 레이트 (통신 속도)
    baudrate=9600,
)


r= sr.Recognizer()

text = '페이지를 넘길게요'

file_name = 'sample.mp3'

tts_ko = gTTS(text = text, lang='ko')

tts_ko.save(file_name)
playsound(file_name)
actions = ['left', 'right']
seq_length = 30

model = load_model('models\model.h5')

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1)


seq = []
action_seq = []

i_pred = 2
sent_time = 0

while cap.isOpened():

    
    current_time = time.time()
    ret, img = cap.read()

    img0 = img.copy()
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]
            
            if conf < 0.9:
                continue               
            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

            cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    if(current_time -sent_time > 6 and i_pred !=2):
        sent_time = current_time
        value_sub = i_pred + 3
        value_main = i_pred
        if(i_pred == 0):
            print("오른쪽으로 넘김")
            py_serial_sub.write(value_sub.to_bytes(1, byteorder='big'))
            py_serial_main.write(i_pred.to_bytes(1, byteorder='big'))
        elif(i_pred == 1):
            print("왼쪽으로 넘김")
            py_serial_sub.write(value_sub.to_bytes(1, byteorder='big'))
            py_serial_main.write(i_pred.to_bytes(1, byteorder='big'))
        i_pred = 2
        print(i_pred)
    if cv2.waitKey(1) == ord('v'):
        with sr.Microphone() as source:
            print('듣고있어요')
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio,language = 'ko')
            print(text)
            if('왼쪽' in text):
                print("왼쪽으로 넘김")
                i_pred = 0
                value_sub = 3
                py_serial_sub.write(value_sub.to_bytes(1, byteorder='big'))
                py_serial_main.write(i_pred.to_bytes(1, byteorder='big'))
                     
            elif('오른쪽' in text):
                i_pred = 1   
                print("오른쪽으로 넘김")
                value_sub = 3
                py_serial_sub.write(value_sub.to_bytes(1, byteorder='big'))
                py_serial_main.write(i_pred.to_bytes(1, byteorder='big'))
        except sr.UnknownValueError:
            print('인식 실패')
        except sr.RequestError:
            print('요청실패')
    # out.write(img0)
    # out2.write(img)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break