import cv2
import numpy as np
import random
import pandas as pd
import math
import glob
import pickle

coef = np.array([[[0.114, 0.587, 0.299]]])


def random_crop(img, size):
    w, h = img.shape[0], img.shape[1]
    rangew = (w - size) // 2
    rangeh = (h - size) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return img[offsetw:offsetw + size, offseth:offseth + size, :]


def center_crop(img, size):
    centerw, centerh = img.shape[0] // 2, img.shape[1] // 2
    halfw, halfh = size // 2, size // 2
    return img[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh, :]


def resize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def random_flip(img, size):
    if np.random.uniform() < 0.5:
        # horizontal_flip
        img = np.asarray(img).swapaxes(1, 0)
        img = img[::-1, ...]
        img = img.swapaxes(0, 1)

    else:
        # vertical_flip
        img = np.asarray(img).swapaxes(0, 0)
        img = img[::-1, ...]
        img = img.swapaxes(0, 0)

    return img


def brightness_aug(img, brightness=0.2):
    alpha = 1.0 + np.random.uniform(-brightness, brightness)
    img *= alpha
    return img


def contrast_aug(img, contrast=0.2):
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img *= alpha
    img += gray
    return img


def saturation_aug(img, saturation=0.4):
    alpha = 1.0 + np.random.uniform(-saturation, saturation)
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    gray *= (1.0 - alpha)
    img *= alpha
    img += gray
    return img


def color_jitter(img):
    lst = [brightness_aug, contrast_aug, saturation_aug]
    random.shuffle(lst)
    for aug in lst:
        img = aug(img)
    return img.astype(np.uint8)


def normalize(img):
    mean_pixel = [103.939, 116.779, 123.68]
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]
    # img = img.transpose((2,0,1))
    # img = np.expand_dims(img, axis=0)
    return img


IMAGES_FOLDER = "/home/venkat/ClothingAttributeDataset/images/"

# preprocess train data
train_df = pd.read_csv("/home/venkat/ClothingAttributeDataset/preprocessed/category_train.csv")
train_imgs = list(train_df["images"])
train_labels = train_df[['shirt', 'sweater', 't-shirt', 'outerwear', 'suit', 'tank_top', 'dress']].values

X_train = []
y_train = []

for i in range(len(train_imgs)):
    img_path = IMAGES_FOLDER + train_imgs[i]
    img = cv2.imread(img_path)
    img_resize = normalize(resize(img, 224))
    img_rf = random_flip(img_resize, 224)
    img_crop = normalize(center_crop(img, 224))
    img_cj = normalize(color_jitter(resize(img, 224).astype(np.float64)))
    X_train += [img_resize, img_rf, img_crop, img_cj]
    temp = [list(train_labels[i]), list(train_labels[i]), list(train_labels[i]), list(train_labels[i])]
    y_train += temp

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

pickle.dump(X_train, open("X_train.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(y_train, open("y_train.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

# preprocess test data
test_df = pd.read_csv("/home/venkat/ClothingAttributeDataset/preprocessed/category_test.csv")
test_imgs = list(test_df["images"])
test_labels = test_df[['shirt', 'sweater', 't-shirt', 'outerwear', 'suit', 'tank_top', 'dress']].values

X_test = []
y_test = []

for i in range(len(test_imgs)):
    img_path = IMAGES_FOLDER + test_imgs[i]
    img = cv2.imread(img_path)
    img_resize = normalize(resize(img, 224))
    X_test += [img_resize]
    temp = [list(test_labels[i])]
    y_test += temp

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

pickle.dump(X_test, open("X_test.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
pickle.dump(y_test, open("y_test.pkl", "wb"), pickle.HIGHEST_PROTOCOL)

print X_train.shape
print X_test.shape
