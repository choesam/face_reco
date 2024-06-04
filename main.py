import cv2
print("debug---------1")
import face_recognition
print("debug---------2")


# 사전 학습할 이미지를 불러옵니다.
image_to_be_matched = face_recognition.load_image_file('aa.jpeg')
print("debug---------3")
name = "Choi"
# 로드된 벡터를 특징 벡터로 인코딩합니다.
image_to_be_matched_encoded = face_recognition.face_encodings(image_to_be_matched)[0]
print(image_to_be_matched_encoded)

# 웹캠을 실행합니다.
webcam = cv2.VideoCapture(0)

# 웹캠 실행한다면 찾을수없다는 메세지와 꺼줍니다.
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# loop through frames
while webcam.isOpened():

    # 웹캠으로 부터 프레임을 읽어 옵니다.
    status, frame = webcam.read()

    # 불러오지 못한다며 해당 경고 메세지를 출력합니다.
    if not status:
        print("Could not read frame")
        exit()

    # face_locations = face_recognition.face_locations(frame) # HoG 기반 얼굴 검출기
    # HOG란 History Of Gradient 의 약자 이미지 경계의 기울기 벡터 크기(magnitude)와 방향(direction)을 히스토그램으로 나타내 계산
    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model="cnn") # CNN 기반 얼굴 검출기

    for face_location in face_locations:
        # 해당하는 이미지의 각 면의 위치를 프린트합니다.
        top, right, bottom, left = face_location
        print("debug---------4")
        print(top, right, bottom, left)

        # 다음과 같이 실제 자신의 얼굴에 엑세스합니다.
        face_image = frame[top:bottom, left:right]
        print("debug---------5")
        print(face_image)

        try:
            print("debug---------6")
            face_encoded = face_recognition.face_encodings(face_image)[0]
            print(face_encoded) # 여기까지만 실행됨.
            print("debug---------7")
            result = face_recognition.compare_faces([image_to_be_matched_encoded], face_encoded, 0.5)
            if result[0] == True:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                Y = top - 10 if top - 10 > 10 else top + 10
                text = name
                cv2.putText(frame, text, (left, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        except:
            pass

    # 화면의 출력문입니다.
    cv2.imshow("detect me", frame)

    # pQ를 누르면 종료됩니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()