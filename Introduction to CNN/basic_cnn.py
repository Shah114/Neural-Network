'''Importing Relevant Libraries'''
import seaborn as sns
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(2019)

'''Loading MNIST Dataset'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

'''Scaling The Data'''
x_train = x_train / 255
x_test = x_test / 255

# flattenning
x_train_flattened = x_train.reshape(len(x_train), 28 * 28)
x_test_flattened = x_test.reshape(len(x_test), 28 * 28)

'''Designing The Neural Network'''
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,), 
                          activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)

'''Confusion Matrix for visualization of predictions'''
y_predict = model.predict(x_test_flattened)
y_predict_labels = [np.argmax(i) for i in y_predict]

cm = tf.math.confusion_matrix(labels=y_test, 
                              predictions=y_predict_labels)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()