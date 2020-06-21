from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

def imaging_preprocessing(path, seed = 1, batch_size = 32):
  data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  #scaling to 0-1
  data_generator = data_generator.flow_from_directory(directory = path, target_size = (256, 256), color_mode='grayscale', class_mode = None, seed = seed)
  return data_generator

def segmentation_preprocessing(path, seed = 1, batch_size = 32):
  data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./127, dtype=np.int32)
  data_generator = data_generator.flow_from_directory(directory = path, target_size = (256, 256), color_mode='grayscale', class_mode=None, seed = seed)
  return data_generator
