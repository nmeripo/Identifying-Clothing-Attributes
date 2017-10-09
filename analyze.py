import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve, auc


test =  pd.read_csv("/home/venkat/ClothingAttributeDataset/preprocessed/category_test.csv")
labels= list(test.columns)
del labels[0]
y_true =  np.asarray(test[labels])
print y_true.shape

num_classes = len(labels)
with open('/home/venkat/y_pred.pkl', 'rb') as f:
        y_pred = pickle.load(f)

y_pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
y_pred = y_pred.argmax(1)
y_true = y_true.argmax(1)


# Micro .. Macro F1 scores
print f1_score(y_true, y_pred, average='micro')
print f1_score(y_true, y_pred, average='macro')


# Plot confusion matrix and normalized confusion matrix
cm = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(cm, index = [i for i in labels],
                  columns = [i for i in labels])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

cm_norm = cm / cm.astype(np.float).sum(axis=0)
df_cm_norm = pd.DataFrame(cm_norm, index = [i for i in labels],
                  columns = [i for i in labels])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm_norm, annot=True)
plt.show()
