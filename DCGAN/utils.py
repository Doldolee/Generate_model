import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display


def mnist_data():
  BUFFER_SIZE = 60000
  BATCH_SIZE = 256

  (train_images, train_labels), (_,_) = tf.keras.datasets.mnist.load_data()
  train_images = train_images.reshape(train_images.shape[0], 28,28,1).astype("float32")
  train_images = (train_images - 127.5) /127.5 # z-score 정규화와 비슷, -1~1사이로 정규화
  # 데이터 배치를 만들고 섞습니다.
  train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

  return train_dataset

  # #테스트
  # for i in train_dataset:
  #   print(i.shape)
  #   break


def generate_and_save_images(model, epoch, test_input):
 
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()