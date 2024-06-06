import cv2
import numpy as np
import pygame
import pyttsx3
import face_recognition
import threading
import time
import os

# 얼굴 데이터를 저장할 딕셔너리
face_data = {}

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
                "voice_path": os.path.join("voices", f"{name}.mp3")
            }
        else:
            print(f"No face detected in {filename}")

# known_face_encodings와 known_face_texts, known_face_voices 리스트 생성
known_face_encodings = [data["encoding"] for data in face_data.values()]
known_face_texts = [name for name in face_data.keys()]
known_face_voices = [data["voice_path"] for data in face_data.values()]

# 웹캠을 실행합니다.
webcam = cv2.VideoCapture(0)

# 웹캠 실행 확인
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# pygame 초기화
pygame.mixer.init()
engine = pyttsx3.init()

# 마지막으로 음성을 출력한 시간을 저장할 변수
last_spoken_time = 0

# 음성을 출력하는 함수 (비동기적으로 실행)
def read_text(name, voice_path):
    if os.path.exists(voice_path):
        pygame.mixer.music.load(voice_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    else:
        engine.say(name)
        engine.runAndWait()

# loop through frames
while webcam.isOpened():
    try:
        # 웹캠으로부터 프레임을 읽어 옵니다.
        status, frame = webcam.read()

        # 불러오지 못한다면 해당 경고 메세지를 출력합니다.
        if not status:
            print("Could not read frame")
            exit()

        # 얼굴 위치와 인코딩을 찾습니다.
        face_locations = face_recognition.face_locations(frame, model="hog")
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # 얼굴 매칭을 수행합니다.
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_texts[best_match_index]
                voice_path = known_face_voices[best_match_index]

                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                current_time = time.time()
                if current_time - last_spoken_time > 3:  # 3초 간격
                    threading.Thread(target=read_text, args=(name, voice_path)).start()
                    last_spoken_time = current_time

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
