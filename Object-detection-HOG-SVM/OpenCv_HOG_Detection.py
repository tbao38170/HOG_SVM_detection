import numpy as np
import cv2
import sys
from glob import glob
import itertools as it

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    # ta ve hinh chu nhat theo may do HOG trả về hình chữ nhật lớn hơn so với vâth phát hiện
    # để có thể đeph hơn ta thu nhỏ hình chữ nhât lại 1 chút
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 0, 255), thickness)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# sử dụng SVM đã đc cài đặt thông qua bộ mô tả HOG đc thư viện OpenCV ( cv )cung câps
# càn dunùng dữ liệu ầu vào hợp lý vì HOG-SVM có giới hạn

# tạo 1 số dữ liệu đa vào dẫn
#path = 'video/istockphoto-1193707800-640_adpp_is.mp4'
#path ='videoplayback.mp4'
#path_1='video/park.mp4'
path ='video/SaltCity2.mp4'
cap= cv2.VideoCapture(path)
#cap= cv2.VideoCapture(0)
#path_Cam = 'http://10.16.21.0:8080/video'
#cap = cv2.VideoCapture(path_Cam)


#đạt cờ để scá định dừng chương trình
#should_stop = False

while True :  # dùng chạy theo thời gian thực

    ret, img = cap.read() # dọc khung hình video hoặc webcam rồi trả về ret
    print(img.shape)
    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
    # mô tả 1 số tham số HOg rồi trả về chúng dới dạng hàm found
    found_filtered = []
    #lọc người
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
            else:
                found_filtered.append(r)
        draw_detections(img, found) # nhận diện người  theo hình hộp chữ nhật hay đóng gói người theo hình hộp giới hạn
        draw_detections(img, found_filtered, 3) #tạo thêm hộp để cải thiện hiển th
        print('loading: %d =>  (%d) : person' % (len(found_filtered), len(found)))
        #số hộp web và số người được ti thấy
    cv2.imshow('img', img) #hiển thị hình ảnh đầu ra
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    #  break
    # if cv2.waitKey(1) == ord('q') or should_stop:
    #     should_stop = True

    if cv2.waitKey(1) == 27:
        #should_stop= True
        break
cap.release()
cv2.destroyAllWindows()
sys.exit()