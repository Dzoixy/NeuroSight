import cv2

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray,(224,224))
    img = cv2.equalizeHist(img)
    return img