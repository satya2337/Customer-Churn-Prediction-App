import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing  import StandardScaler, LabelEncoder

import pickle
import os
import datetime


data = pd.read_csv("churn_dataset.csv")

data.head()
data = data.drop(['RowNumber' , 'CustomerId' , 'Surname'] , axis=1)
 
label_encoder_gender = LabelEncoder()

data['Gender'] = label_encoder_gender.fit_transform(data['Gender']) 

from sklearn.preprocessing import OneHotEncoder

onehot_encoder_geo = OneHotEncoder()

geo_encoder = onehot_encoder_geo.fit_transform(data[['Geography']]).toarray()

onehot_encoder_geo.get_feature_names_out(['Geography'])

geo_encoder_df = pd.DataFrame(
    geo_encoder,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

data=pd.concat(
    [data.drop('Geography', axis=1) , geo_encoder_df],
    axis=1
)

with open('label_encoder_gender.pkl' , 'wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_geo , file)


X = data.drop('Exited' , axis = 1)

y = data['Exited']

X_train ,X_test, y_train , y_test = train_test_split(
    X,y, test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


with open('scaler.pkl','wb') as file:
    pickle.dump(scaler, file)
    


import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense 

from tensorflow.keras.callbacks import EarlyStopping , TensorBoard

import datetime

model = Sequential([
    Dense(64 , activation='relu', input_shape=(X_train.shape[1],)),
    
    Dense(32 , activation='relu'),
    
    Dense(1 , activation='sigmoid')
])    

model.summary()
    
opt = tf.keras.optimizers.Adam(learning_rate=0.01) 

loss = tf.keras.losses.BinaryCrossentropy()


model.compile(
    optimizer = opt,
    loss = "binary_crossentropy",
    metrics=['accuracy']
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorBoard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

early_stopping_callback = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs = 100,
    callbacks=[tensorBoard_callback , early_stopping_callback]
)

model.save('ann_churn_model.h5')

#load_ext tensorBoard
#tensorBoard --logdir logs/fit