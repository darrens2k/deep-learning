# Assignment 3 - MMAI 5500 - Deep Learning
# Darren Singh
# 216236275

# pep8 formatted by via flake8

import os
import cv2
from PIL import Image
from glob import glob
import numpy as np
import keras
from keras import layers
from keras.models import load_model

# File paths
model_path = 'model.h5'
video_path = 'assignment3_video.avi'
frames_path = 'frames'

# These booleans control what parts of the code are run
# In its default state only the deliverables will be run
# Change it if all of the starter/prep code is to be run as well
run_prep_code = False

# Deliverables at bottom of file

if run_prep_code:

    # function to convert video to frames
    def convert_video_to_images(img_folder, filename='assignment3_video.avi'):
        """
        Converts the video file (assignment3_video.avi) to JPEG images.
        Once the video has been converted to images, then this function doesn't
        need to be run again.
        Arguments
        ---------
        filename : (string) file name (absolute or relative path) of video
        file.
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

    # extract frames from video, only needs to be done once
    convert_video_to_images(frames_path)

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
    flat_images, image_list = load_images(frames_path)
    print(flat_images.shape, len(image_list))

    # building simplest autoencoder (fully connected neural layer)

    # This is the size of our encoded representations
    encoding_dim = 32

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

    # random shuffle of x_train before training
    x_train = np.array(image_list)
    np.random.shuffle(x_train)

    # train autoencoder
    autoencoder.fit(x_train, x_train,
                    epochs=30,
                    batch_size=128,
                    shuffle=True)

    # save model after training
    autoencoder.save(model_path)

# Deliverables

# load model
model = load_model(model_path)


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

    # Since using convolutional structure, needed to change the input shape
    # Input layer does not take a flattened image
    frame = frame.reshape((1, 44, 60, 3))
    loss = model.evaluate(frame, frame, verbose=0)

    if loss > 0.5165:
        anomaly = True

    return anomaly
