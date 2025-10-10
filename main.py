# type: ignore
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(trainging_images, trainging_labels), (test_images, test_labels) = datasets.cifar10.load_data()

trainging_images, test_images = trainging_images / 255.0, test_images / 255.0
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(trainging_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[trainging_labels[i][0]])
# plt.show()

trainging_images = trainging_images[:20000]
trainging_labels = trainging_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]

# model =models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10 , activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(trainging_images, trainging_labels, epochs=10, validation_data=(test_images, test_labels))

# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# print('\nTest accuracy:', test_acc)
# print('Test loss:', test_loss)

# model.save('image_classification.keras')

model =models.load_model('image_classification.keras')

img =cv.imread('airplane.jpg')
img =cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img , cmap=plt.cm.binary)
prediction = model.predict(np.array([img]) / 255.0)
index =np.argmax(prediction)
print("This is a " + class_names[index])
plt.show()
