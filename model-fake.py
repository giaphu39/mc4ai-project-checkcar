import os
import torch
import torchvision.transforms as transforms
import pickle
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import pygame
import pygame.mixer as mixer
import pygame
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

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

pretrained_model = PretrainModel()
label_dict = read_training_data_label()
data_dict = read_training_data()
X = []
y = []
for filename, features in data_dict.items():
    label = label_dict.get(filename)
    if label is not None:
        X.append(features)
        y.append(label)

model = train_model(X, y)
save_model(model)
loaded_model = load_model()
mixer.init()
mixer.music.load(r"warning.mp3")
mixer.music.set_volume(0.7)
webcam=cv2.VideoCapture(0)
while True:
    ret,frame=webcam.read()
    if ret==True:
        input_image = Image.fromarray(frame)
        feature_vector = pretrained_model.get_feature(input_image)
        prediction = loaded_model.predict([feature_vector])
        height, width = frame.shape[:2]
        color = (0, 255, 0) 
        x1 = width // 2 - 200
        y1 = height // 2 - 200
        x2 = width // 2 + 200
        y2 = height // 2 + 200
        if prediction == 1:
            color = (0, 0, 255) 
            if not mixer.music.get_busy():
                mixer.music.play(fade_ms=3000)
                print("detected car")
        else:
            color = (0, 255, 0)
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.imshow("cars",frame)
    key=cv2.waitKey(60) & 0xFF
    if prediction == 1 and (not mixer.music.get_busy()):
        mixer.music.play(fade_ms=3000)
        print("detected car")
    if key==ord("q"):
        break
