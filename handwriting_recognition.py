import pandas as pd
import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
import tensorflow as tf

### IMPORTING THE DATASET
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

### RESHAPING PIXELS IN IMAGES TO BE 2D
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

### SAMPLING THE FIRST 100 ITEMS
X_train = X_train[:1000, :]
y_train = y_train[:1000]
X_test = X_test[:100, :]
y_test = y_test[:100]

### CREATING A MACHINE LEARNING MODEL
model = svm.SVC()
model.fit(X_train, y_train)

### PREDICTIONS
y_pred = model.predict(X_test)

### SELECTED INDEX TO PREDICT
index_to_compare = 16
 
### DISPLAYING THE ANSWER AND PREDICTION
title = 'True: ' + str(y_test[index_to_compare]) + ', Prediction: ' + str(y_pred[index_to_compare])
 
### DISPLAYING THE IMAGE
plt.title(title)
plt.imshow(X_test[index_to_compare].reshape(28,28), cmap='gray')
plt.grid(None)
plt.axis('off')
plt.show()

### CALCULATING ACCURACY
acc = metrics.accuracy_score(y_test, y_pred)
print('\nAccuracy: ', acc)

### STORING DIGITS IN DATASET
digits = pd.DataFrame.from_dict(y_train)


###----------ACCURACY TRACKING VISUALISATIONS


### BAR PLOT TO SEE DATA DISTRIBUTION WITH SEABORN
ax = sns.countplot(x=0, data=digits)
 
### LABELING THE BAR GRAPH
ax.set_title("Distribution of Digit Images in Test Set")
ax.set(xlabel='Digit')
ax.set(ylabel='Count')

### SHOWING THE BAR GRAPH
plt.show()

### CREATING A CONFUSION MATRIX TO TRACK ERRORS
cm = metrics.confusion_matrix(y_test, y_pred)
 
### CREATE A 9 x 6 PLOT
ax = plt.subplots(figsize=(9, 6))
 
### CREATE A HEATMAP
sns.heatmap(cm, annot=True)
 
### CHANGE AXIS LABELS
ax[1].title.set_text("SVC Prediction Accuracy")
ax[1].set_xlabel("Predicted Digit")
ax[1].set_ylabel("True Digit")
 
### SHOWING THE HEATMAP
plt.show()
