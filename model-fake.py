import os
import torch
import torchvision.transforms as transforms
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.linear_model import LogisticRegression
class PretrainModel:
    def __init__(self):
        self.model = torch.load("models/pretrained.pkl")
        self.model.eval()
        self.transformer =  transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def get_feature(self, pil_image):
        image_tensor = self.transformer(pil_image)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            feature = self.model(image_tensor).squeeze().detach().cpu().numpy()
        return feature
def read_training_data_label():
    with open("data/image_label.pkl", "rb") as f:
        return pickle.load(f)
def read_training_data():
    features_dict = {}
    folder_path = 'data/images'
    pretrained = PretrainModel()
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        pil_image = Image.open(img_path).convert("RGB")
        features_dict[filename] = pretrained.get_feature(pil_image)
    return features_dict

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def save_model(model):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

# Load the logistic regression model
with open("model.pkl", "rb") as f:
  clf = pickle.load(f)
# Load the pre-trained model
pretrain_model = PretrainModel()
# Open a camera or video feed
cap = cv2.VideoCapture(0)
while True:
  # Capture frame-by-frame
  ret, frame = cap.read()
  # Convert the image to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Detect cars in the frame using a car detector (e.g. Haar cascade)
  detected_cars = []
  # ...
  # For each detected car, extract the feature vectors using the pre-trained model
  # and use the logistic regression model to predict the presence of a car
  cars_present = []
  for car in detected_cars:
    # Preprocess the image
    x1, y1, w, h = car
    x2, y2 = x1 + w, y1 + h
    car_image = gray[y1:y2, x1:x2]
    # Extract the image features using the pre-trained model
    features = pretrain_model.extract_features(car_image)
    # Make a prediction using the logistic regression model
    prediction = clf.predict([features])
    if prediction[0] == 1:
      # Car is present
      cars_present.append(car)
  # Draw bounding boxes around the detected cars
  for car in cars_present:
    x1, y1, w, h = car
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
  # Display the resulting frame
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
# Release the capture
cap.release()
cv2.destroyAllWindows()