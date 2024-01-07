from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import sklearn.externals
from sklearn.externals import joblib
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob
import matplotlib
import scipy

#xac dinhj tham so cua HOG
#thay doi chung thanh huong = 8 , pixels  tren mo o = (16,16) , chia o tren mo khoi bang (2,2) cho HOg de chay
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = 0.3

#tao ra cua so truot tren anh
def sliding_window(image, stepSize,windowSize):  # hinh anh dau vao , so pixel can loai bo va tao windowSize la kich thuoc cho cua so thuc te
    # cho truot ngang qua hinh anh
    for y in range(0, image.shape[0],stepSize):
        for x in range(0, image.shape[1], stepSize):
            #truot tren toa do x va y
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

#upload la mo hinh
model = joblib.load('model/model.npy')

scale = 0
detections = []
# doc thu muc anh
img = cv2.imread("image/positive/10.jpg")

# thay doi khich thuoc anh neu anh qua lon
img = cv2.resize(img,(500, 700))

#xac dinh thich thuoc cua so truot lai gan bang kich thuoc anh
(winW, winH) = (64, 128)
windowSize = (winW, winH)
downscale = 1.5
# ap dung cua so truot
for resized in pyramid_gaussian(img, downscale=1.5):
    # lap qua tung lop cua hinh anh
    for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH) ):
        # neu cua so truot ko daps ung dc kich thuoc cua sua so mong muon thif ta  nen bo qua no
        if window.shape[0] != winH or window.shape[1] != winW:
        #can dam bao cua so truot dap ung dc  kich thuoc toi thieu
            continue
        if window.ndim == 3 and window.shape[2] == 3:
            window = color.rgb2gray(window)
        else:
            print("cua so khong phai mau rgb")
            continue
        fds = hog(window, orientations, pixels_per_cell, cells_per_block,block_norm='L2')
        # trich xuat HOG dac trung tung cua so da chup
        fds = fds.reshape(1, -1)  # dinh hinh lai hinh anh de ra hinh bong
        pred = model.predict(fds)  #su dung SV< de dua ra du doan
        if pred == 1:
            if model.decision_function(fds) > 0.6:  #dat xac xuat du doan = 0.6
                print("Phát hiện:: Vị trí -> ({}, {})".format(x, y))
                print("tỉ lệ ->  {} | điểm tin cậy {} \n".format(scale, model.decision_function(fds)))
                detections.append((int(x * (downscale ** scale)), int(y * (downscale ** scale)), model.decision_function(fds),
                     int(windowSize[0] * (downscale ** scale)),  # tao ra ds cac du doan dc tim thay
                     int(windowSize[1] * (downscale ** scale))))
    scale += 1

clone = resized.copy()
for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness=2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])  #
sc = [score[0] for (x, y, score, w, h) in detections]
print("Diem tin cay duoc phat hien : ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
# đoạn mã trên tạo một khung giới hạn thô trước khi sử dụng NMS
# đoạn mã bên dưới tạo một khung giới hạn sau khi sử dụng nms trên các phát hiện
# cv2.imshow ở đúng vị trí này (vì python là thủ tục nên nó sẽ đi qua từng dòng mã).

for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
cv2.imshow("phat hien ra NMS", img)
save_dir ='image/save'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path= os.path.join(save_dir, 'img_1.jpg')
cv2.imwrite(save_path,img)
k =  cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'): # nhan s phím s de lưu
    # cv2.imwrite('image/save/1.jpg', img)
    cv2.destroyAllWindows()