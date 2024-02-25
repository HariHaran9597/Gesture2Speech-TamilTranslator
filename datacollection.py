import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Mapping from English alphabet to Tamil characters
alphabet_to_tamil = {
    'A': 'அ',
    'B': 'ப',
    'C': 'ச',
    # Add more mappings as needed
}

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Offset and size parameters for cropping and resizing the hand region
offset = 20
imgSize = 300

# Folder to save the captured images
folder = "Data/C"

# Counter for the captured images
counter = 0

while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    if hands:
        # Take the first detected hand
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white image for better visibility of hand gesture
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region with an offset
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # Resize the cropped region to fit into a square of size imgSize x imgSize
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (imgSize, wCal))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)

            # Ensure that imgResize fits into imgWhite correctly
            imgWhite[:, wGap:wCal + wGap] = imgResize[:imgSize, :wCal]
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)

            # Ensure that imgResize fits into imgWhite correctly
            imgWhite[hGap:hCal + hGap, :] = imgResize[:hCal, :]

        # Display the cropped hand region and the white image
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the original frame with hand landmarks
    cv2.imshow("Image", img)

    # Check for keypress 's' to save the image
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        # Save the white image with a timestamp as the filename
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
