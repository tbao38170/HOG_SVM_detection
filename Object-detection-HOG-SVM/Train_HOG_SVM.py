import os
import numpy as np
from PIL import Image
from numpy import *
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import sklearn.externals
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# xác định tham số trích xuất đặc trưng Bước 1
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# tạo đường dẫn đến hình ảnh
pos_im_path = r"image/positive" # link vô ảnh người
# tạo link cho không pahỉ người
neg_im_path = r"image/negative"

# read the image files:
pos_im_listing = os.listdir(pos_im_path) # đọc file ảnh người
neg_im_listing = os.listdir(neg_im_path) # dọc file ảnh không phải người
num_pos_samples = size(pos_im_listing) # neue toongr số hình ảnh
num_neg_samples = size(neg_im_listing)
print('so anh trong pos %s ' % num_pos_samples)  # in giá trị số mẫu trong pos
print(num_neg_samples) # in giá trị số mẫu trong neg
data = []
labels = []


# #tính toán HOG và gắn thêm nha

for file in pos_im_listing:  # đọc ừng biến trong tệp pos
    img = Image.open(pos_im_path + '\\' + file)  # mở file
    img = img.resize((64,128))
    gray = img.convert('L')  # chuyển đổi từ RGB màu qua ảnh xám
    # băt đa tính HOG đặc trưng
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2',feature_vector=True)
    data.append(fd)
    labels.append(1)

# làm tương tự cho neg
for file in neg_im_listing:
    img = Image.open(neg_im_path + '\\' + file)
    img = img.resize((64,128))
    gray = img.convert('L')
 # tính HOg đặc trưng cho từng điểm tiêu cực
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(0)
#mã hoá các nhãn đã đặt chuyển đổi chúng về số nguyên
le = LabelEncoder()
labels = le.fit_transform(labels)


# ta nên phân vùng d leiẹu cho train và test , sử dụng khoảng 80%
print(" Dang thuc hien phan chia training va testing...") # im ra đang thực hiện
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.20, random_state=42)
#train_test_split để chia tập dữ liệu thành traning và testing
# np.array(data) là tập dữ liệu đặc trưng HOG , labels là  nhãn tương ứng với các hình ảnh.
#test_size=0.20 xác định tỷ lệ tập kiểm tra là 20% so với tổng số dữ liệu.
#random_state=42 đảm bảo kết quả chia tập dữ liệu nhất quán mỗi khi chạy.
#kết quar se duco luu vo trainDate, testData, trainlabels , testLabels


print(" Training  :  cho cac phan loai tuyen tinh SVM !")
model = LinearSVC(dual='auto') # tao ra 1 doi tuong LinearSVC ,  lay trong thu vien scikit-learn
model.fit(trainData, trainLabels)
print(" testing : danh gia phan loai tren doi tuong thu nghiem khong phai nguoi !")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))
#  sử dụng hàm classification_report từ thư viện scikit-learn ể tính toán vàđánh giá hiệu suất của mô hình
joblib.dump(model, 'model/model.npy')

