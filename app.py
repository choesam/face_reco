#엑셀파일로 저장하는 코드

import cv2
import numpy as np
import pygame
import pyttsx3
import face_recognition
import threading
import time
import os
import serial
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# 얼굴 데이터를 저장할 딕셔너리
face_data = {}

# 로그 파일 초기화
log_file = "face_recognition_log.csv"

# 로그 파일에 헤더 작성
with open(log_file, 'w') as f:
    f.write("time,name\n")

# faces 디렉토리에서 이미지 파일을 읽어 얼굴 데이터를 학습합니다.
for filename in os.listdir("faces"):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join("faces", filename)
        face_image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding) > 0:
            face_data[name] = {
                "encoding": face_encoding[0],
                "image_path": image_path,
                "voice_path": os.path.join("voices", f"{name}.mp3"),
                "text": f"{name}님, 안녕하세요"
            }
        else:
            print(f"No face detected in {filename}")

# known_face_encodings와 known_face_texts, known_face_voices 리스트 생성
known_face_encodings = [data["encoding"] for data in face_data.values()]
known_face_texts = [data["text"] for data in face_data.values()]
known_face_voices = [data["voice_path"] for data in face_data.values()]

# pygame 초기화
pygame.mixer.init()
engine = pyttsx3.init()

# 마이크로비트 직렬 포트 설정
# SERIAL_PORT = 'COM3'  # 적절한 포트로 설정
# ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)

def send_command(command):
    ser.write((command + '\n').encode())
    time.sleep(0.1)

def set_servo_angle(angle):
    send_command(str(angle))

def stop_servo():
    send_command('stop')

# 마지막으로 음성을 출력한 시간을 저장할 변수
last_spoken_time = 0

# 음성을 출력하는 함수 (비동기적으로 실행)
def read_text(text, voice_path):
    if os.path.exists(voice_path):
        pygame.mixer.music.load(voice_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    else:
        engine.say(text)
        engine.runAndWait()

# 한글 폰트 설정
font_path = "fonts/NanumGothic.ttf"  # 시스템에 설치된 한글 폰트 경로
if not os.path.exists(font_path):
    raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {font_path}")
font = ImageFont.truetype(font_path, 20)

def draw_text(image, text, position):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    draw.text(position, text, font=font, fill=(255, 0, 0, 0))
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# 웹캠을 실행합니다.
webcam = cv2.VideoCapture(0)

# 웹캠 실행 확인
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# loop through frames
while webcam.isOpened():
    try:
        # 웹캠으로부터 프레임을 읽어 옵니다.
        status, frame = webcam.read()

        # 불러오지 못한다면 해당 경고 메시지를 출력합니다.
        if not status:
            print("Could not read frame")
            break

        # 얼굴 위치와 인코딩을 찾습니다.
        face_locations = face_recognition.face_locations(frame, model="hog")
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        max_width = 60
        best_match_index = None
        best_face_location = None

        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            width = right - left  # 얼굴의 가로 크기

            if width > max_width:
                # 얼굴 매칭을 수행합니다.
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                current_best_match_index = np.argmin(face_distances)

                if matches[current_best_match_index]:
                    max_width = width
                    best_match_index = current_best_match_index
                    best_face_location = face_location

        if best_match_index is not None and best_face_location is not None:
            name = list(face_data.keys())[best_match_index]
            voice_path = known_face_voices[best_match_index]
            text = known_face_texts[best_match_index]

            top, right, bottom, left = best_face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            frame = draw_text(frame, name, (left, top - 25))

            current_time = time.time()
            if current_time - last_spoken_time > 2:  # 3초 간격
                threading.Thread(target=read_text, args=(text, voice_path)).start()
                # set_servo_angle(45)  # 서보 모터 제어 명령 예시
                last_spoken_time = current_time

                # 얼굴 탐지 기록 저장
                detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_file, 'a') as f:
                    f.write(f"{detection_time},{name}\n")

        # 화면에 프레임을 출력합니다.
        cv2.imshow("Face Recognition", frame)

        # 'q'를 누르면 종료됩니다.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"Error: {e}")

# 자원 해제
webcam.release()
cv2.destroyAllWindows()
# ser.close()
