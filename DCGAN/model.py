import tensorflow as tf

class generator_model(tf.keras.Model):
  def __init__(self):
    super(generator_model, self).__init__()
    
    # input값은 랜덤 시드로 (batch, 100)차원으로 들어온다. batch는 tf가 control하므로 input_shape는 batch를 제외하고 형태를 잡아주면 된다.
    self.dense = tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,))
    self.batch1 = tf.keras.layers.BatchNormalization()
    self.batch2 = tf.keras.layers.BatchNormalization()
    self.batch3 = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.LeakyReLU()

    self.reshape = tf.keras.layers.Reshape((7,7,256))

    self.conv2dt_1 = tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias = False)
    self.conv2dt_2 = tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False)
    self.conv2dt_3 = tf.keras.layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False)


  def call(self, x):
    x = self.dense(x)
    x = self.batch1(x)
    x = self.relu(x)
    x = self.reshape(x)
    # assert x.shape == (None, 7, 7, 256) # 주목: 배치사이즈로 None이 주어집니다.
    x = self.conv2dt_1(x)
    # assert x.shape == (None, 7, 7, 128)
    x = self.batch2(x)
    x = self.relu(x)
    x = self.conv2dt_2(x)
    # assert x.shape == (None, 14, 14, 64)
    x = self.batch3(x)
    x = self.relu(x)
    logits = self.conv2dt_3(x)
    # assert logits.shape == (None, 28, 28, 1)

    return logits

class discriminator_model(tf.keras.Model):
  def __init__(self):
    super(discriminator_model, self).__init__()

    self.conv_1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1])
    self.conv_2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
    self.relu = tf.keras.layers.LeakyReLU()
    self.dropout1 = tf.keras.layers.Dropout(0.3)
    self.dropout2 = tf.keras.layers.Dropout(0.3)
    self.flatten = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(1)

  def call(self, x):
    x = self.conv_1(x)
    x = self.relu(x)
    x = self.dropout1(x)
    x = self.conv_2(x)
    x = self.relu(x)
    x = self.dropout2(x)
    x = self.flatten(x)
    logits = self.dense(x)

    return logits


discriminator = discriminator_model()