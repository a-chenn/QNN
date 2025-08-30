
#sources : Deep Learning AI; AIEngineering(YT); CodeEmporium(YT)




import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses, Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, MaxPooling2D

(train_images, _), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

input_shape = (28, 28, 1)

inputs = Input(shape=input_shape)
x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(inputs) # no padding, and tensoring it to all the inputs

model = Model(inputs=inputs, outputs=x)

train_images_pooled = model.predict(train_images) # eeunning it through the model so it can poooled it down
test_images_pooled = model.predict(test_images)

train_images_pooled, val_images_pooled = train_images_pooled[:-10000], train_images_pooled[-10000:]

# what a 1x10 vector
latent_dim = 8

class Autoencoder(Model):
    #okk _init_ is a speacil python constructor, we need when we call a new instance of class is created
    #helps us customize the layers of keras model
    #self is used to refer to the current instance 
    #ex. car class(make, model) --> self.make and self.model to get the attribute
    def __init__(self, latent_dim):
        #auto-encoder is current class(sub class)
        #super- (i'm calling a method fromt the parent class)- in this case the parent class is(tf.Keras.mdoodel)
        super(Autoencoder, self).__init__() #wanna accces soeme methods form Keras; dont know we want _int_-*


        #self.latent_dim allows us to work with even if outsie the __init_ method

        self.latent_dim = latent_dim
         
         #keras.Sequential makes a linear stack of layers
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            #Dense is layer where every neuron is connected the neruon in the anotehr layer
            #Relu is an acitivation function!! * understand before moving on*
            layers.Dense(latent_dim, activation='relu'),
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(14*14, activation='sigmoid'),  # Output size should match the pooled image size
            layers.Reshape((14, 14,1)), # remmeber this the orginal input
        ])
    
    def call(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


autoencoder = Autoencoder(latent_dim)

#adam optimizer?!(* read into it)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(train_images_pooled, train_images_pooled, epochs=10, shuffle=True, validation_data=(val_images_pooled, val_images_pooled))





#jst wanted to see how it would like

encoded_imgs = autoencoder.encoder(tf.convert_to_tensor(test_images_pooled, dtype=tf.float32))  # Encode images
decoded_imgs = autoencoder.decoder(encoded_imgs)    # Decode images

decoded_imgs_np = decoded_imgs.numpy()
'''
# Plot results
fig, axes = plt.subplots(2, 10, figsize=(15, 4))

for i in range(10):
    axes[0, i].imshow(test_images_pooled[i].reshape(14, 14), cmap='gray')
    axes[0, i].axis('off')  # Hide axes

for i in range(10):
    axes[1, i].imshow(decoded_imgs_np[i].reshape(14, 14), cmap='gray')
    axes[1, i].axis('off')  # Hide axes

#plt.show()
'''
#just want to get the encoder part, disgaring the decoder part of the alogorithm
encoded_imgs = autoencoder.encoder.predict(test_images_pooled)

np.savetxt("images.txt", np.concatenate([test_labels.reshape((len(test_labels), 1)), encoded_imgs], axis = 1))
'''
for i in range(10):
    print("Image" + str(i+1))
    print(encoded_imgs[i]) 
    print() #want some spcae
'''