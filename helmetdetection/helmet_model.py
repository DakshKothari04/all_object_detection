import numpy as np
import cv2
import os
import random 
import pickle

Dir = r'images/train'
CATEGORIES = ['nothelmet','hemet']

img_size = 24
data = []

for category in CATEGORIES:
    folder = os.path.join(Dir,category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.cvtColor(img_arr,cv2.COLOR_BGR2GRAY)
        img_arr = cv2.resize(img_arr,(img_size,img_size),1)
        data.append([img_arr,label])

random.shuffle(data)
X= []
Y= []

for features,label in data:
    X.append(features)
    Y.append(label)
    
X = np.array(X)
Y = np.array(Y)

pickle.dump(X,open('X.pkl','wb'))
pickle.dump(Y,open('Y.pkl','wb'))

X = X/255
img_rows,img_cols = 24,24
X =X.reshape(X.shape[0],img_rows,img_cols,1)

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

model = Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape = X.shape[1:]))
model.add(MaxPooling2D(1,1))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(1,1))

model.add(Conv2D(64 , (3,3) , activation = 'relu'))
model.add(MaxPooling2D((1,1)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))

# compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy' , metrics = ['accuracy'])
# fit X , Y to the model to see accuracy of model:
model.fit(X, Y, epochs = 5 , validation_split = 0.1 , batch_size = 32)

model.save("helmetmodel.h5")