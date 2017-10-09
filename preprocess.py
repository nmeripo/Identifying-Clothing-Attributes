import pandas as pd
import numpy as np
import glob
import scipy.io
import shutil
from tqdm import tqdm
import os


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


ROOT = "/home/venkat/ClothingAttributeDataset/"
LABELS = "/home/venkat/ClothingAttributeDataset/labels/"
PREPROCESS = "/home/venkat/ClothingAttributeDataset/preprocessed/"

if not os.path.exists(PREPROCESS):
    os.makedirs(PREPROCESS)

val = ["No", "Yes"]
data_colors = {'black': val, 'blue': val, 'brown': val, 'cyan': val, 'gray': val, 'green': val, 'purple': val,
               'red': val, 'white': val, 'yellow': val, 'purple': val, 'many_colors': val}

data_pattern = {'pattern_floral': val, 'pattern_graphics': val, 'pattern_plaid': val,
                'pattern_solid': val, 'pattern_spot': val, 'pattern_stripe': val}

data_binary = {'collar': val, 'gender': ["male", "female"], 'necktie': val,
               'placket': val, 'skin_exposure': ["low", "high"], 'scarf': val}

data_multi = {'sleevelength': ["no", "short", "long"], 'neckline': ["v-shape", "round", "other"],
              'category': ["shirt", "sweater", "t-shirt", "outerwear", "suit", "tank_top", "dress"]}

data = merge_dicts(data_colors, data_binary, data_pattern)

category_df = pd.DataFrame()

for filename in glob.iglob(LABELS + '*.mat'):
    feature_name = filename.split("/")[-1].split(".")[0][:-3]

    if feature_name == "category":
        labels = data_multi[feature_name]
        mat = scipy.io.loadmat(filename)['GT'].flatten()
        category_df = pd.get_dummies(mat, prefix="category")
        category_df.columns = labels
        category_df.insert(0, "images", category_df.index.map(lambda val: "{:06d}.jpg".format(val + 1)))
        category_df = category_df[~np.isnan(mat)]

# train-test split randomly
msk = np.random.rand(len(category_df)) < 0.8
train = category_df[msk]
test = category_df[~msk]

# Data Percentage for each category
for key in data_multi['category']:
    print key, round(100 * category_df[key].value_counts()[1]/ float(category_df.shape[0]), 2)

train.to_csv(PREPROCESS + "category_train" + ".csv", index=False)
test.to_csv(PREPROCESS + "category_test" + ".csv", index=False)


# For Keras ImageGenerator - Flow from Directory
"""
train_label_map = {}

for item in data_multi['category']:
    train_label_map[item] = list(train.loc[train[item] == 1]["images"])

test_label_map = {}
for item in data_multi['category']:
    test_label_map[item] = list(test.loc[test[item] == 1]["images"])

label_cols = list(train.columns)
del label_cols[0]
y_train = train[label_cols].values
y_test = test[label_cols].values

copy_path_train = ROOT + "category_train/"
copy_path_test = ROOT + "category_test/"

if not os.path.exists(copy_path_train):
    os.makedirs(copy_path_train)

if not os.path.exists(copy_path_test):
    os.makedirs(copy_path_test)

for key in train_label_map.keys():
    class_path = copy_path_train + key

    if not os.path.exists(class_path):
        os.makedirs(class_path)
    img_paths = train_label_map[key]

    for path in img_paths:
        src_path = "/home/venkat/ClothingAttributeDataset/images/" + path
        copy_path = class_path + "/" + path
        shutil.copyfile(src_path, copy_path)

for key in test_label_map.keys():
    class_path = copy_path_test + key
    
    if not os.path.exists(class_path):
        os.makedirs(class_path)
    img_paths = test_label_map[key]
    
    for path in img_paths:
        src_path = "/home/venkat/ClothingAttributeDataset/images/" + path
        copy_path = class_path + "/" + path
        shutil.copyfile(src_path, copy_path)
"""
