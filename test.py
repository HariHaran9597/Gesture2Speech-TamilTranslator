import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from gtts import gTTS
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Mapping from English alphabet to Tamil characters
alphabet_to_tamil = {
    'A': 'அ',
    'B': 'ப',
    'C': 'ச',
    # Add more mappings as needed
}

# Function to convert predicted index to English label and Tamil characters
def convert_to_tamil(prediction, index, labels):
    english_label = labels[index]
    tamil_output = alphabet_to_tamil.get(english_label, english_label)
    return english_label, tamil_output

# Load a font that supports Tamil characters
font_path = "Noto_Sans_Tamil/NotoSansTamil-VariableFont_wdth,wght.ttf"  # Replace with your Tamil font path
font_size = 30  # Adjust font size as needed
font = ImageFont.truetype(font_path, font_size)

# Load classifier model and labels
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Specify image path
img_path = r"C:\Users\HariHaran\Pictures\Camera Roll\WIN_20240225_12_07_16_Pro.jpg"  # Replace with your image path

# Read the image
img = cv2.imread(img_path)

# Initialize hand detector
hand_detector = HandDetector(maxHands=1)

# Define possible English labels based on your classifier output
labels = ['A', 'B', 'C']  # Update with your specific labels

# Detect hands in the image and draw hand landmarks
hands, img = hand_detector.findHands(img, draw=True)

if hands:
    # Take the first detected hand
    hand = hands[0]
    x, y, w, h = hand['bbox']

    # Crop and resize hand region
    imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]
    imgResize = cv2.resize(imgCrop, (300, 300))

    # Get prediction and index from classifier
    prediction, index = classifier.getPrediction(imgResize, draw=False)

    # Convert prediction to English label and Tamil output
    english_label, tamil_output = convert_to_tamil(prediction, index, labels)

    print(f"Predicted Label: {english_label}")
    print(f"Tamil Output: {tamil_output}")

    # Draw English and Tamil labels on the image
    cv2.putText(img, f"Prediction: {english_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Blue for English

    # Convert the image to PIL format for text drawing
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 70), f"Tamil Output: {tamil_output}", font=font, fill=(0, 255, 0))  # Green for Tamil
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Convert Tamil output to speech and save it as an audio file
    tts_tamil = gTTS(text=tamil_output, lang='ta')
    tts_tamil.save("output_tamil.mp3")
    os.system("start output_tamil.mp3")

    # Draw bounding box around detected hand
    img = cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(0)

else:
    print("No hands detected in the image.")
