# Assignment 3 - MMAI 5500 - Deep Learning
# Darren Singh
# 216236275

import os
import cv2
from PIL import Image
from glob import glob
import numpy as np
import keras
from keras import layers
from keras.models import load_model
from matplotlib import pyplot as plt

### These booleans control what parts of the code are run
# In its default state only the deliverables will be run
# Change it if all of the starter/prep code is to be run as well
run_prep_code = False
run_deliverables = True

# File paths


if run_prep_code:

    # function to convert video to frames
    def convert_video_to_images(img_folder, filename='assignment3_video.avi'):
        """
        Converts the video file (assignment3_video.avi) to JPEG images.
        Once the video has been converted to images, then this function doesn't
        need to be run again.
        Arguments
        ---------
        filename : (string) file name (absolute or relative path) of video file.
        img_folder : (string) folder where the video frames will be
        stored as JPEG images.
        """

        # Make the img_folder if it doesn't exist.'
        try:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
        except OSError:
            print('Error')

        # Make sure that the abscense/prescence of path
        # separator doesn't throw an error.
        img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
        # Instantiate the video object.
        video = cv2.VideoCapture(filename)

        # Check if the video is opened successfully
        if not video.isOpened():
            print("Error opening video file")

        i = 0
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                im_fname = f'{img_folder}frame{i:0>4}.jpg'
                print('Captured...', im_fname)
                cv2.imwrite(im_fname, frame)
                i += 1
            else:
                break
        video.release()
        cv2.destroyAllWindows()
        if i:
            print(f'Video converted\n{i} images written to {img_folder}')

    ### Uncomment this step to extract frames from video
    # extract frames from video, only needs to be done once
    # convert_video_to_images('frames')

    # function to load the images
    def load_images(img_dir, im_width=60, im_height=44):
        """
        2
        Reads, resizes and normalizes the extracted image frames from a folder.
        The images are returned both as a Numpy array of flattened images
        (i.e. the images with the 3-d shape (im_width, im_height, num_channels)
        are reshaped into the 1-d shape (im_width x im_height x num_channels))
        and a list with the images with their original number of dimensions
        suitable for display.
        Arguments
        ---------
        img_dir : (string) the directory where the images are stored.
        im_width : (int) The desired width of the image.
        The default value works well.
        im_height : (int) The desired height of the image.
        The default value works well.
        Returns
        X : (numpy.array) An array of the flattened images.
        images : (list) A list of the resized images.
        """

        images = []
        fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
        fnames.sort()

        for fname in fnames:
            im = Image.open(fname)
            # resize the image to im_width and im_height.
            im_array = np.array(im.resize((im_width, im_height)))
            # Convert uint8 to decimal and normalize to 0 - 1.
            images.append(im_array.astype(np.float32) / 255.)
            # Close the PIL image once converted and stored.
            im.close()

        # Flatten the images to a single vector
        X = np.array(images).reshape(-1, np.prod(images[0].shape))

        return X, images

    # call function to load images from frames
    flat_images, image_list = load_images('frames')
    print(flat_images.shape, len(image_list))

    # building simplest autoencoder (fully connected neural layer)

    # This is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # # This is our input image
    # input_img = keras.Input(shape=(7920,))
    # # "encoded" is the encoded representation of the input
    # encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # # "decoded" is the lossy reconstruction of the input
    # decoded = layers.Dense(7920, activation='sigmoid')(encoded)

    # # This model maps an input to its reconstruction
    # autoencoder = keras.Model(input_img, decoded)

    # # This model maps an input to its encoded representation
    # encoder = keras.Model(input_img, encoded)

    # # This is our encoded (32-dimensional) input
    # encoded_input = keras.Input(shape=(encoding_dim,))
    # # Retrieve the last layer of the autoencoder model
    # decoder_layer = autoencoder.layers[-1]
    # # Create the decoder model
    # decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    # # define loss
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    input_img = keras.Input(shape=(44, 60, 3))

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # create training and test sets, 95% for training and 5% for testing
    # there is a total of 1050 images

    ### maybe do a random shuffle of data before splitting into training and testing set
    np.random.shuffle(flat_images)
    percentage_split = int(0.7 * len(image_list))
    x_train = np.array(image_list)
    np.random.shuffle(x_train)
    x_test = np.array(flat_images[percentage_split:])

    # train autoencoder
    autoencoder.fit(x_train, x_train,
                    epochs=250,
                    batch_size=128,
                    shuffle=True)

    # save model after training
    autoencoder.save('model.h5')

if run_deliverables:

    # load model
    model = load_model('model.h5')

    def predict(frame):
        """
        Argument
        --------
        frame : Video frame with shape == (44, 60, 3) and dtype == float.
        Return
        anomaly : A boolean indicating whether the frame is an anomaly or not.
        ------
        """

        # define return variable
        anomaly = False

        # get reconstruction loss
        ### Since using convolutional structure, needed to change the input shape
        # Input layer does not take a flattened image
        frame = frame.reshape((1,44,60,3))
        loss = model.evaluate(frame, frame, verbose=0)

        if loss > 0.513:
            anomaly = True
            
        # Your fancy computations here!!
        return anomaly, loss
    flat_images, image_list = load_images('frames')

    for im in image_list[690:980]:
        out, loss = predict(im)
        if not out:
            plt.imshow(im.reshape(44, 60, 3))
            plt.gray()
            plt.show()
            print(loss)