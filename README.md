<h1 align="center">
    <a href="https://pt-br.reactjs.org/">🔗 Behavioural Cloning Applied to Self-Driving Car on a Simulated Track using the Nvidia model</a>
</h1>
<p align="center">🚀 Step by step of how to use my project</p>

This project is an implementation of the Nvidia model for self-driving car. It has an objective to train a model to drive a car autonomously on a simulated track. The ability of the model to drive the car is learned from cloning the behaviour of a human drive. The training data is gotten from examples of a real human driving in the simulator, then fed into a convolutional neural network which learns what to do (calculate the steering angle) for every frame in the simulation.

If you wish to learn more about the model proposed by Nvidia, you can read about it at the [Nvidia Developer Blog](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/) or read the [published paper about the model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

For more information about this project, feel free to read the article Behavioural Cloning Applied to Self-Driving Car on a Simulated Track using the Nvidia model (available on this GitHub).

This project is divided in 4 parts:
1. [The driving simulator](#the-driving-simulator)
2. [Creating the Convolutional Neural Network (CNN) for training](#creating-the-convolutional-neural-network)
3. [Connecting the driving simulator with the CNN](#connecting-the-driving-simulator)
4. Model testing and validating


# The driving simulator

For this project we'll use the driving simulator developed by Udacity for its [Udacity's Self-Driving Car Nanodegree](https://udacity.com/drive). It has two modes: 
* Training mode: used to gather data about the driving pattern
* Autonomous mode: used to test and validate the model created by the CNN

To download the simulator, [click here](https://github.com/udacity/self-driving-car-sim), and select the Version 1. It is avaialable for Windows, Linux and Mac.
To gather the training data, just start the simulator, select the Track 1 (we will use the Track 2 to validate the model generalization) and click in Training mode.

<img width="805" alt="Schermata 2021-10-08 alle 10 23 06" src="https://user-images.githubusercontent.com/19311371/136565283-0d608807-7ac0-47b3-8ea3-604883ca3bf1.png">

Before start driving, remember to click on the Record Button (upright corner). Remember: a student is only as good as his teacher! Try to drive the best way you can. Avoid getting out of the road and try to stay at the middle of the lane most of the time. Since its a racing track, complete at least 2 laps and then turn around and complete at least 2 laps on the other way. This will avoid having a biased dataset for training (turning left mos of the time, for example).

When you complete all the laps, click to stop recording (upright corner) and close the simulator. The dataset that will be used for training will be saved on your Desktop. It consists with pictures from 3 cameras (located on the right side, left side and center of the car) as well information about the speed, steering angle and throtle. 

# Creating the Convolutional Neural Network


Since we are using the Google Colab plataform, there is no need to prepare an environment to execute the Neural Network. Simply [click here](https://colab.research.google.com/drive/1fecPOfW5oZC3WviqG5GRXs6URtlkAobd?authuser=1&hl=pt-BR) to open the Google Colab Notebok containing the untrained neural network.
First thing we need to do is change the runtime type to use a GPU as hardware accelerator. To do so, select Runtime and then Select Runtime Type. Under Hardware Accelerator, choose GPU in the dropdown menu and then click in save. That's all the configuration we need to do.

Note: to execute every step just click on the play button located at the top left corner of each box that contain code.

#### Step 1
First we need to install a library called imgaug. This python library helps you with augmenting images for your machine learning projects. It converts a set of input images into a new, much larger set of slightly altered images. Just run this Step and wait for its completion.

#### Step 2
We need to import all the libraries used to train and validate the neural network. Also, we need to import the libraries used to manipulate the files and plot graphs. After executing Step 2 you should get an output like this:

```json
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 11233807447978002051
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 11344216064
locality {
  bus_id: 1
  links {
  }
}
incarnation: 8972007624555681685
physical_device_desc: "device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7"
]
```
If you are using the correct runtime type, you shoudl see a GPU listed as a device_type.


#### Step 3
We need to download the dataset containing the images for training, validation and testing. After downloading, we decompress the file and delete every hidden file that might be created during the compressing. This step might take a while depending on yout internet speed. A

The code used is:

```python
import pathlib
!gdown --id 1YRf2QfJoIIxE11YMVxGhcr0DpSVYoJnp
!unzip /content/imgCarro.zip #Descompactando o dataset
```

#### Step 4
Defining the name of the directory containing the dataset. In addition to the dataset images, we have to import a CSV file that contains information about each image. For every time one of the car's cameras captured an image, we have the following information:
* Name of the photo that was captured by the camera in the middle of the car
* Name of the photo that was captured by the camera on the left side of the car
* Name of the photo that was captured by the camera on the right side of the car
* Steering wheel angle
* Accelerator pedal angle
* Angle of the accelerator pedal if backing up
* Normalized car speed. Sine 0 the car stopped and 1 the car at its maximum speed (set via software).

The code used is:

```python
datadir = 'imgCarro'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
pd.set_option('display.max_colwidth', -1)
data.head()
```

After executing Step 4 you should get an output like this:

```json

                                  center	                                                          left	                                                                    right	                       steering	throttle	reverse	speed
0	C:\Users\Amer\Desktop\new_track\IMG\center_2018_07_16_17_11_43_382.jpg	C:\Users\Amer\Desktop\new_track\IMG\left_2018_07_16_17_11_43_382.jpg	C:\Users\Amer\Desktop\new_track\IMG\right_2018_07_16_17_11_43_382.jpg	0.0	0.0	0.0	0.649786
1	C:\Users\Amer\Desktop\new_track\IMG\center_2018_07_16_17_11_43_670.jpg	C:\Users\Amer\Desktop\new_track\IMG\left_2018_07_16_17_11_43_670.jpg	C:\Users\Amer\Desktop\new_track\IMG\right_2018_07_16_17_11_43_670.jpg	0.0	0.0	0.0	0.627942
2	C:\Users\Amer\Desktop\new_track\IMG\center_2018_07_16_17_11_43_724.jpg	C:\Users\Amer\Desktop\new_track\IMG\left_2018_07_16_17_11_43_724.jpg	C:\Users\Amer\Desktop\new_track\IMG\right_2018_07_16_17_11_43_724.jpg	0.0	0.0	0.0	0.622910
3	C:\Users\Amer\Desktop\new_track\IMG\center_2018_07_16_17_11_43_792.jpg	C:\Users\Amer\Desktop\new_track\IMG\left_2018_07_16_17_11_43_792.jpg	C:\Users\Amer\Desktop\new_track\IMG\right_2018_07_16_17_11_43_792.jpg	0.0	0.0	0.0	0.619162
4	C:\Users\Amer\Desktop\new_track\IMG\center_2018_07_16_17_11_43_860.jpg	C:\Users\Amer\Desktop\new_track\IMG\left_2018_07_16_17_11_43_860.jpg	C:\Users\Amer\Desktop\new_track\IMG\right_2018_07_16_17_11_43_860.jpg	0.0	0.0	0.0	0.615438
```

#### Step 5

Plotting the histogram of the data obtained in the training dataset. This step is necessary to verify that the data is not skewed in any position. In it we can see that most of the time the car has the steering wheel in a straight position (angle 0). This is normal while driving but for training the neural network can make her learn that she should stay with the steering wheel in a straight position.

The code used is:
```python
def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()
num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1]+ bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
```
You should see the following output:

<img width="383" alt="Schermata 2021-10-08 alle 10 57 10" src="https://user-images.githubusercontent.com/19311371/136569782-f7ac0566-8915-4374-aad0-9b910657c81d.png">

#### Step 6

To get around the problem exposed in Step 5, let's remove some of the dataset entries in an attempt to increase the exposure of the other steering wheel angles. After removing some of the entries, we re-plot the histogram to see if the data is less skewed. 2590 samples were removed, remaining 1463 samples for learning.
From the histogram we can see that the dataset is now more balanced. Even though we are most of the time with the steering wheel in the straight position, the other steering wheel angles now represent a significant portion of the learning process.

The code used is:
```python
print('total de dados:', len(data))
remove_list = []
for j in range(num_bins):
  list_ = []
  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)
 
print('removidos:', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('permaneceram:', len(data))
 
hist, _ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
```
You should see the following output:

<img width="375" alt="Schermata 2021-10-08 alle 10 57 40" src="https://user-images.githubusercontent.com/19311371/136569958-def5aa46-0f6c-4813-be81-a3071115e079.png">

#### Step 7

Preparing the dataset to be split between training dataset and validation dataset. An object is created containing the 3 photos captured by the cameras, along with acceleration, reverse, speed and steering wheel position information. As what we are trying to learn is the position of the steering wheel according to the position in which the car is, the defined function returns the directory that contains the images and the position of the steering wheel angle in each of the photos.

The code used is:
```python
print(data.iloc[1])
def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
    image_path.append(os.path.join(datadir,left.strip()))
    steering.append(float(indexed_data[3])+0.15)
    image_path.append(os.path.join(datadir,right.strip()))
    steering.append(float(indexed_data[3])-0.15)
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings
 
image_paths, steerings = load_img_steering(datadir + '/IMG', data)
```
You should see the following output:
````json
center      center_2018_07_16_17_11_43_998.jpg
left        left_2018_07_16_17_11_43_998.jpg  
right       right_2018_07_16_17_11_43_998.jpg 
steering    0                                 
throttle    0                                 
reverse     0                                 
speed       0.606834                          
Name: 6, dtype: object
````

#### Step 8

Defining validation training datasets. For this, 80% of the images were used for training and 20% of the images for validation.

The code used is:
```python
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Amostras de treinamento: {}\nAmostras de validação: {}'.format(len(X_train), len(X_valid)))
```
You should see the following output:
````json
Training Samples: 3511
Valid Samples: 878
````

#### Step 9

Just for viewing if the training and validation dataset have similar characteristics, the histogram containing the position of the steering wheel angle was plotted for both training and validation. We can see that both datasets are similar. This is necessary because, if they were very different, the training validation process could be considered bad despite the training being done correctly.

The code used is:
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Treinamento')
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title('Validação')
```
You should see the following output:

<img width="713" alt="Schermata 2021-10-08 alle 11 03 26" src="https://user-images.githubusercontent.com/19311371/136570837-f8747c93-4d6f-496f-bc18-74745efae325.png">

#### Step 10

Defining the functions for Image Augmentation. These functions will be used in the next step to generate new images from the images that exist in the dataset. 4 functions were defined to change the original image generating new images, namely:

* Enlarge the image
* Apply Montion Blur to the image
* Change image brightness
* Mirror image

The code used is:
```python
def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.3))
  image = zoom.augment_image(image)
  return image
image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)
 
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
 
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed Image')

def pan(image):
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
  image = pan.augment_image(image)
  return image

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)
 
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
 
axs[1].imshow(panned_image)
axs[1].set_title('Panned Image')
def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image
image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
brightness_altered_image = img_random_brightness(original_image)
 
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
 
axs[1].imshow(brightness_altered_image)
axs[1].set_title('Brightness altered image ')

def img_random_flip(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle
random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]
 
 
original_image = mpimg.imread(image)
flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)
 
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
 
axs[0].imshow(original_image)
axs[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))
 
axs[1].imshow(flipped_image)
axs[1].set_title('Flipped Image - ' + 'Steering Angle:' + str(flipped_steering_angle))

```
You should see the following output:

<img width="1069" alt="Schermata 2021-10-08 alle 11 05 38" src="https://user-images.githubusercontent.com/19311371/136571125-92bebe0d-d285-4683-b4a2-babb7ba3cbce.png">

#### Step 11

Defining a function that will randomly call one of the Image Augmentation methods defined above. After calling this function our training dataset will be expanded.

The code used is:
```python
def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
      image = pan(image)
    if np.random.rand() < 0.5:
      image = zoom(image)
    if np.random.rand() < 0.5:
      image = img_random_brightness(image)
    if np.random.rand() < 0.5:
      image, steering_angle = img_random_flip(image, steering_angle)
    
    return image, steering_angle

ncol = 2
nrow = 10
 
fig, axs = plt.subplots(nrow, ncol, figsize=(15, 50))
fig.tight_layout()
 
for i in range(10):
  randnum = random.randint(0, len(image_paths) - 1)
  random_image = image_paths[randnum]
  random_steering = steerings[randnum]
    
  original_image = mpimg.imread(random_image)
  augmented_image, steering = random_augment(random_image, random_steering)
    
  axs[i][0].imshow(original_image)
  axs[i][0].set_title("Original Image")
  
  axs[i][1].imshow(augmented_image)
  axs[i][1].set_title("Augmented Image")
```

#### Step 12
The images used in the dataset have many elements that are not useful for training, such as:
* the top of the photo has sky and trees
* the bottom of the photo shows a part of the car
* the sides of the photo have part of the environment beyond the track

All these items mentioned above do not contribute positively to the training. The important thing for learning is to know the limits of the track as we are interested in defining the position of the car's steering wheel. For this, the image will go through a pre-processing to remove these elements from the image. Additionally, the image color can also be changed to a scale that facilitates training. Then the image's color scheme is changed, a Gaussian filter is applied, the image is resized and then normalized. Thus, the image that will be used in training has only the elements important for training and has a size substantially smaller than the original image.

The code used is:
````python
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
image = image_paths[100]
original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(original_image)
 
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(preprocessed_image)
axs[1].set_title('Preprocessed Image')
````

You should see the following output:

<img width="1071" alt="Schermata 2021-10-08 alle 11 07 31" src="https://user-images.githubusercontent.com/19311371/136571613-d655c5d5-0d3a-4481-95b1-75d1c0023a4f.png">

#### Step 13

Creating batches with images and position of the steering wheel for training.

The code used is:
````python
def batch_generator(image_paths, steering_ang, batch_size, istraining):
  
  while True:
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)
      
      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      
      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))  
````

#### Step 14

Defining training and validation batches. Shows an example of an image used in the training dataset and an image used in the validation dataset.

The code used is:
````python
x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
 
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
 
axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training Image')
 
axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation Image')
````

You should get the following output:

<img width="1062" alt="Schermata 2021-10-08 alle 11 10 14" src="https://user-images.githubusercontent.com/19311371/136571983-ef015520-35cc-45cf-9f9e-ca476e78c362.png">

#### Step 15

Defining the Nvidia model.

The code used is:
````python
def nvidia_model():
 
  model = Sequential()
  
  model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), input_shape=(66,200,3),activation='elu'))
  
  model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='elu'))
  model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
  model.add(Conv2D(64, kernel_size=(3,3), activation='elu'))
  model.add(Dropout(0.5))
  
  
  model.add(Flatten())
  model.add(Dense(100, activation='elu'))
  model.add(Dropout(0.5))
  
  
  model.add(Dense(50, activation='elu'))
  model.add(Dense(10, activation ='elu'))
  model.add(Dense(1))
  
  
  optimizer= Adam(learning_rate=1e-3)
  model.compile(loss='mse', optimizer=optimizer)
  
  return model
````

#### Step 16

Training the Nvidia model.

The code used is:
````python
model = nvidia_model()
print(model.summary())
history = model.fit(batch_generator(X_train, y_train, 100, 1),
                                  steps_per_epoch=300, 
                                  epochs=10,
                                  validation_data=batch_generator(X_valid, y_valid, 100, 0),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle = 1)
````

You should get the following output:

````json
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
dropout (Dropout)            (None, 1, 18, 64)         0         
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0         
_________________________________________________________________
dense (Dense)                (None, 100)               115300    
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_2 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/10
300/300 [==============================] - 254s 751ms/step - loss: 0.1329 - val_loss: 0.0666
Epoch 2/10
300/300 [==============================] - 223s 744ms/step - loss: 0.0798 - val_loss: 0.0576
Epoch 3/10
300/300 [==============================] - 219s 733ms/step - loss: 0.0733 - val_loss: 0.0530
Epoch 4/10
300/300 [==============================] - 223s 745ms/step - loss: 0.0713 - val_loss: 0.0484
Epoch 5/10
300/300 [==============================] - 220s 735ms/step - loss: 0.0689 - val_loss: 0.0465
Epoch 6/10
300/300 [==============================] - 221s 738ms/step - loss: 0.0665 - val_loss: 0.0431
Epoch 7/10
300/300 [==============================] - 220s 736ms/step - loss: 0.0649 - val_loss: 0.0471
Epoch 8/10
300/300 [==============================] - 220s 737ms/step - loss: 0.0603 - val_loss: 0.0444
Epoch 9/10
300/300 [==============================] - 223s 744ms/step - loss: 0.0618 - val_loss: 0.0484
Epoch 10/10
300/300 [==============================] - 217s 726ms/step - loss: 0.0594 - val_loss: 0.0411
`````
#### Step 17

The result of the training. Don't forget to execute this step since it will download the trained model to your computer. It will be necessary for the test and validation of the model using the driving simulator.

The code used is:
````python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
model.save('model.h5')
from google.colab import files
files.download('model.h5')
````
You should get the following output (plus the downloaded model.h5 file):

<img width="380" alt="Schermata 2021-10-08 alle 11 13 45" src="https://user-images.githubusercontent.com/19311371/136572696-7b043ffe-a31b-4e7d-a623-974043439fbc.png">

# Connecting the driving simulator

To connect the driving simulator to the trained model, first we need to install some libraries. If you are using Mac OS or Linux (Ubuntu 20.04 LTS), run the following command at the terminal:

````
pip install eventlet numpy flask keras tensorflow venv pillow opencv-python python-socketio
````
If you encounter any errors trying to install the tensorflow library, try the following:
````
conda install tensorflow
````
After all those libraries are installed, we need to install one more to go:

````
pip install 'h5py=2.10.0' --force-reinstall
````

Now we have everything we need to connect the driving simulator to the trained model. Download the files automatico.py and model.h5 and put them in the same folder. The file automatico.py is responsible for creating a socket that will connect with the driving simulator and, when selected autonomous mode, analyze every frame from the driving simulator using the trained model to decide the steering angle.

The code inside the automatico.py is:

````python
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()

app = Flask(__name__) #'__main__'
speed_limit = 10
def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    steering_angle = float(model.predict(image))
    throttle = 1.0 - speed/speed_limit
    print('{} {} {}'.format(steering_angle, throttle, speed))
    send_control(steering_angle, throttle)



@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

````

As you can notice, there is a variable called speed_limit. You can set this value from 0 to 30 (miles per hours). After you set the desired speed limit, just run the script above on the terminal:

````
python3 automatico.py
````

Wait until the Web Server Gateway Interface (WSGI) is ready. You should see the following:

````
(9643) wsgi starting up on http://0.0.0.0:4567
````

Then open the driving simualtor, selecte the desired track and click on Autonomous Drive. You can see a video of every step of this process, [here](https://www.youtube.com/watch?v=2i512pD1Llk).
