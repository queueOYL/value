import tensorflow as tf

mnist=tf.keras.datasets.mnist
(x_train,t_train),(x_test,t_test)=mnist.load_data()#读取数据
x_train,x_test = x_train/255.0 , x_test/255.0 #初始化图像的256个色度于0—1区间上

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
     
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),#损失函数
    metrics=['accuracy']
)

model.fit(x_train, t_train, batch_size=60, epochs=10, validation_data=(x_test,t_test), validation_freq=2)
model.summary()



