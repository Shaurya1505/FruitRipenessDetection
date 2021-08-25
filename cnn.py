#keras libraries and packages



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D

#1-initializing CNN
classifier = Sequential()

#2-adding 1st set of Convolution layer and Pooling layer
classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

#3-adding 2nd set of convolution layer and polling layer
classifier.add(Conv2D(32,(3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


#4-Flattening of layers
classifier.add(Flatten())

#5-Full Connection (Creating Artificial Neural Network)

classifier.add(Dense(units=32,activation = 'relu'))

classifier.add(Dense(units=64,activation = 'relu'))

classifier.add(Dense(units=128,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))

classifier.add(Dense(units=6,activation = 'softmax'))

#6-Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#7-Fitting CNN to images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, # To rescaling the image in range of [0,1]
                                   shear_range = 0.2, # To randomly shear the images 
                                   zoom_range = 0.2, # To randomly zoom the images
                                   horizontal_flip = True) #  for randomly flipping half of the images horizontally 

test_datagen = ImageDataGenerator(rescale = 1./255)
print("\nTraining the data...\n")
training_set = train_datagen.flow_from_directory('train',
                                                target_size=(64,64),
                                                batch_size=12, #Total no. of batches
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size=(64,64),
                                            batch_size=12,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch=len(training_set), # Total training images
                         
                         epochs = 20, # Total no. of epochs
                         validation_data = test_set,
                         validation_steps = len(test_set)) # Total testing images

#8-saving model 

classifier.save("model.h5")