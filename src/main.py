import cv2
import sys

faceCascade = cv2.CascadeClassifier("Haar Cascade/haarcascade_frontalface_default.xml")

# Đọc hình ảnh để phân tích
image = cv2.imread("images/ava.jpg")
# Tạo bức ảnh xám từ ảnh trên
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tìm khuôn mặt
faces = faceCascade.detectMultiScale(
    grayImage,
    scaleFactor  = 1.1,
    minNeighbors = 5,
)

print("Found {0} faces!".format(len(faces)))

# Vẽ khung vuông xung quanh khuôn mặt để xác định đây là khuôn mặt
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
