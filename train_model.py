import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, AveragePooling2D, ZeroPadding2D, MaxPooling2D, Conv2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras import applications
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
import sys

# read labels

train = pd.read_csv("/home/gopal/venkat/ClothingAttributeDataset/preprocessed/category_train.csv")
test =  pd.read_csv("/home/gopal/venkat/ClothingAttributeDataset/preprocessed/category_test.csv")

label_cols = list(train.columns)
del label_cols[0]
y_train = train[label_cols].values
y_test = test[label_cols].values

print(y_train.shape)
print(y_test.shape)


# dimensions of our images.
img_width, img_height = 224, 224
img_rows, img_cols = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '/home/gopal/venkat/ClothingAttributeDataset/category_train'
validation_data_dir = '/home/gopal/venkat/ClothingAttributeDataset/category_test'
nb_train_samples = y_train.shape[0]
nb_validation_samples = y_test.shape[0]
epochs = 50
batch_size = 8



def vgg16_model(X_train, y_train, X_test, y_test, img_rows =224, img_cols=224, img_channel=3, num_classes=7):

    base_model = applications.VGG16(weights='imagenet', include_top=False,
                                    input_shape=(img_rows, img_cols, img_channel))

    # extract max pool-4 layer
    for i in xrange(4):
        base_model.layers.pop()

    inp = base_model.input
    out = base_model.layers[-1].output
    mod_model = Model(inp, out)

    print(mod_model.layers[-1])
    print mod_model.output_shape
    add_model = Sequential()
    add_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=mod_model.output_shape[1:]))
    add_model.add(BatchNormalization())
    #add_model.add(MaxPooling2D(pool_size=(2, 2)))
    add_model.add(Dropout(0.1))

    add_model.add(Conv2D(32, (3, 3), activation='relu'))
    add_model.add(BatchNormalization())
    #add_model.add(MaxPooling2D(pool_size=(2, 2)))
    #add_model.add(Dropout(0.2))

    add_model.add(GlobalAveragePooling2D())
    #add_model.add(Flatten())
    #add_model.add(Dropout(0.2))
    #add_model.add(Dense(64, activation='relu'))


    add_model.add(Dense(num_classes, activation='softmax'))

    print add_model.summary()

    final_model = Model(inputs=mod_model.input, outputs=add_model(mod_model.output))
    #final_model.load_weights("final_model.h5")
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    history = final_model.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_test, y_test), shuffle=True)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    """
    # Average augmented data predictions for each test sample
    pred_list = []
    print X_test.shape
    for idx in range(0, X_test.shape[0]):
        if (idx + 1) % 4 == 0:
            print idx
            pred = final_model.predict(X_test[:idx])
            pred_list.append(pred.mean(0))

    preds = np.asarray(pred_list)
    """
    preds = final_model.predict(X_test)
    print(preds.shape)
    pickle.dump(preds, open("y_pred.pkl", "wb"))

    # serialize model to JSON
    model_json = final_model.to_json()
    with open("final_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    final_model.save_weights("final_model.h5")
    print("Saved model to disk")

    # later...

    # load json and create model
    json_file = open('final_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()


with open('./X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('./X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('./y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('./y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)


vgg16_model(X_train, y_train, X_test, y_test)


