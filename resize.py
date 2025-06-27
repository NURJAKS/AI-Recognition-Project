import os
import cv2
import numpy as np
import random
import shutil

source_dir = 'data'
aug_dir = 'data_aug'
target_size = (64, 64)

def clear_aug_dir():
    if os.path.exists(aug_dir):
        shutil.rmtree(aug_dir)
    os.makedirs(aug_dir)

def is_blurry(img, threshold=100):
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

def center_and_crop(img, size):
    h, w = img.shape[:2]
    top = max((h - size[1]) // 2, 0)
    left = max((w - size[0]) // 2, 0)
    return img[top:top+size[1], left:left+size[0]]

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        img = img[y:y+h, x:x+w]

    img = center_and_crop(cv2.resize(img, target_size), target_size)

    if is_blurry(img):
        return None  # skip blurry

    return img

def augment(img):
    augmented = []

    # Яркость и контраст
    for alpha in [0.9, 1.1]:
        for beta in [-10, 10]:
            bright = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            augmented.append(bright)

    # CLAHE (локальная нормализация)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    augmented.append(clahe_img)

    # Отражение
    flip = cv2.flip(img, 1)
    augmented.append(flip)

    # Поворот
    angle = random.choice([-10, 10])
    M = cv2.getRotationMatrix2D((target_size[0]//2, target_size[1]//2), angle, 1)
    rotated = cv2.warpAffine(img, M, target_size)
    augmented.append(rotated)

    return augmented

clear_aug_dir()

for label in os.listdir(source_dir):
    class_dir = os.path.join(source_dir, label)
    if not os.path.isdir(class_dir):
        continue

    save_dir = os.path.join(aug_dir, label)
    os.makedirs(save_dir, exist_ok=True)

    for i, img_name in enumerate(os.listdir(class_dir)):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = preprocess(img)
        if img is None:
            continue

        # Save base image
        base_name = f"{label}_{i}.jpg"
        cv2.imwrite(os.path.join(save_dir, base_name), img)

        # Save augmentations
        for j, aug in enumerate(augment(img)):
            aug_name = f"{label}_{i}_aug{j}.jpg"
            cv2.imwrite(os.path.join(save_dir, aug_name), aug)
