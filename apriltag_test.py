import cv2
import numpy as np
from apriltag_test import apriltag

imagepath = 'tags/tag36_11_00000.png'
image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
detector = apriltag("tag36h11")

detections = detector.detect(image)