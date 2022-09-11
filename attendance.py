import sqlite3
import face_recognition
import cv2
import csv
import numpy as np
import face_recognition
import os
from datetime import datetime


def slice_date_time(_item):
    print(_item.split(':')[0])
    return _item.split(':')[0]

def is_attended(items):
    data = []
    f = open('attendance.csv')
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
    # import ipdb; ipdb.set_trace()
    print(f'All csv: {data}')
    print(f'Our items: {items}')
    for i in data:
        i[1] = slice_date_time(i[1])
    if items in data:
        print("itams in data")
        return True
    return False



def write_attendance_to_csv(items):
    new_item = items.copy()
    items[1] = slice_date_time(items[1])
    if is_attended(items):
        return False
    with open('attendance.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(new_item)
    return True


face_sets = set()
name_sets = []
con = sqlite3.connect('./db.sqlite3')

cur = con.cursor()
record = [ i for i in cur.execute('SELECT * FROM student_management_app_students')]
photo = [f'.{i[2]}' for i in record]
id_for_name = [f'{i[7]}' for i in record]
names = []
# get name from id_for_name
for i in id_for_name:
    for row in cur.execute(f'SELECT username FROM student_management_app_customuser WHERE id == {i}'):
        names.append(row[0])

 
def findEncodings(images):
    encodeList = []
    for img in images:
        pic = face_recognition.load_image_file(img)
        encode = face_recognition.face_encodings(pic)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(photo)
print("Encoding Complete")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


final = []


while True:
    today = datetime.today()
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    img_small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    img_small = cv2.flip(img_small, 1)
    rgb_small_frame = img_small[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    encodeCurFrame = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for encodeFace, faceLoc in zip(encodeCurFrame, face_locations):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.4)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        print(matches)
        d = None
        for i in matches:
            if i == True:
                d = matches.index(True)
                name = names[int(d)]
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                current_date = today.strftime("%d%m%Y")
                current_time = today.strftime("%H%M%S")
                combine = current_date + ':' + current_time
                print(combine)
                write_attendance_to_csv([name, combine])
        
    cv2.imshow("WebCam", frame)
    if cv2.waitKey(5) == ord('q'):
        break

cv2.destroyAllWindows()
