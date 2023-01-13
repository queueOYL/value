import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
mnist=tf.keras.datasets.mnist

(x_train,t_train),(x_test,t_test)=mnist.load_data()#读取数据
t_train = to_categorical(t_train,10)
t_test = to_categorical(t_test,10)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train = x_train.reshape([-1,28,28,1])
x_test = x_test.reshape([-1,28,28,1])


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",
    metrics="acc"
)

model.fit(x_train, t_train, batch_size=60, epochs=10, validation_data=(x_test,t_test), validation_freq=5)
model.summary()

model.save('DenseModel.h5')
