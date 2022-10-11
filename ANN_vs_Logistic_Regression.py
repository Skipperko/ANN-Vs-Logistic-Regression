# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
tf.__version__

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

X_LR = dataset.iloc[:, [2,3]].values
y_LR = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(X_LR, y_LR, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train_LR = sc.fit_transform(X_train_LR)
X_test_LR = sc.transform(X_test_LR)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train_LR, y_train_LR)

# print(classifier.predict(sc.transform([[30,87000]])))

# Predicting the Test results
y_pred_LR = classifier.predict(X_test_LR)
print(np.concatenate((y_pred_LR.reshape(len(y_pred_LR),1), y_test_LR.reshape(len(y_test_LR),1)),1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test_LR, y_pred_LR)
print(cm)
accuracy_score(y_test_LR, y_pred_LR)

# Visualising the Training set results
X_set_LR, y_set_LR = X_train_LR, y_train_LR
X1_LR, X2_LR = np.meshgrid(np.arange(start = X_set_LR[:, 0].min() -1,
                               stop = X_set_LR[:, 0].max() + 1,
                               step= 0.01),np.arange(start = X_set_LR[:, 1].min() -1,
                                                     stop = X_set_LR[:, 1].max() + 1,
                                                     step = 0.01))
plt.contourf(X1_LR, X2_LR, classifier.predict(np.array([X1_LR.ravel(), X2_LR.ravel()]).T).reshape(X1_LR.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1_LR.min(), X1_LR.max())
plt.xlim(X2_LR.min(), X2_LR.max())
for i, j in enumerate(np.unique(y_set_LR)):
    plt.scatter(X_set_LR[y_set_LR == j, 0], X_set_LR[y_set_LR == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
X_set_LR, y_set_LR = X_test_LR, y_test_LR
X1_LR, X2_LR = np.meshgrid(np.arange(start = X_set_LR[:, 0].min() -1,
                               stop = X_set_LR[:, 0].max() + 1,
                               step = 0.01),np.arange(start = X_set_LR[:, 1].min() -1,
                                                      stop = X_set_LR[:, 1].max() + 1,
                                                      step = 0.01))
plt.contourf(X1_LR, X2_LR, classifier.predict(np.array([X1_LR.ravel(), X2_LR.ravel()]).T).reshape(X1_LR.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1_LR.min(), X1_LR.max())
plt.xlim(X2_LR.min(), X2_LR.max())
for i, j in enumerate(np.unique(y_set_LR)):
    plt.scatter(X_set_LR[y_set_LR == j, 0], X_set_LR[y_set_LR == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#####################################################################################

X_ANN = dataset.iloc[:, [2,3]].values
y_ANN = dataset.iloc[:, 4].values

# Encoding categorical data
le = LabelEncoder()
X_ANN[:, 0] = le.fit_transform(X_ANN[:, 0])

# Feature Scaling
sc = StandardScaler()
X_ANN = sc.fit_transform(X_ANN)

# Splitting the dataset into the Training set and Test set
X_train_ANN, X_test_ANN, y_train_ANN, y_test_ANN = train_test_split(X_ANN, y_ANN, test_size = 0.2, random_state = 0)

# Initialising the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu',input_dim=2))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
ann1 = ann.fit(X_train_ANN, y_train_ANN, validation_data = (X_test_ANN, y_test_ANN),  batch_size = 32, epochs = 100)

plt.plot(ann1.history['accuracy'])
plt.plot(ann1.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(ann1.history['loss'])
plt.plot(ann1.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Predicting the Test set results
y_pred_ANN = ann.predict(X_test_ANN)
y_pred_ANN = (y_pred_ANN > 0.5)
print(np.concatenate((y_pred_ANN.reshape(len(y_pred_ANN),1), y_test_ANN.reshape(len(y_test_ANN),1)),1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test_ANN, y_pred_ANN)
print(cm)
accuracy_score(y_test_ANN, y_pred_ANN)