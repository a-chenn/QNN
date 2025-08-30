import numpy as np # to manipulate arrays
from PIL import Image # to read images
from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# # Function to convert an image to a binary array
##just want to see what it looks liike
'''
plt.imshow(train_images[0], cmap='gray') # pulling the first image
plt.title(f"Label: {train_labels[0]}")
plt.show()
'''


pooled_image = np.zeros((14, 14)) # 14x14 intlizaed with images
first_image = train_images[0, :, : ] 
for i in range(27): # rows
    for j in range(27): # column #j=0
        max_value = max(first_image[i, j], first_image[i+1,j], first_image[i,j+1], first_image[i+1, j+1]) #using the 2x2 max pooling method
        pooled_image[i//2, j//2] = max_value 
        j = j+2 # only colums is chaning
    i = i+2 # only rows is changing

plt.imshow(pooled_image, cmap='gray')
plt.show()
