import tensorflow as tf


def convLayer(input, k, slope=0.2, stride=2, reuse=False, is_training=True, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights_shape = shape=[4, 4, input.get_shape()[3], k]
    W_var = tf.get_variable("W_var", weights_shape,
      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))
    conv = tf.nn.conv2d(input, W_var, strides=[1, stride, stride, 1], padding='SAME')
    normalized = instance_norm(conv)
    output = tf.maximum(slope*normalized, normalized)###########leakyRelu
    return output



def lastLayer(input, reuse=False, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights_shape = [4, 4, input.get_shape()[3], 1]
    W_var = tf.get_variable("W_var", weights_shape,
      initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))
    bias_shape = [1]
    b_var = tf.get_variable("b_var", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, W_var, strides=[1, 1, 1, 1], padding='SAME')
    output = conv + b_var
    return output


class Discriminator:
  def __init__(self, name, is_training):
    self.name = name
    self.is_training = is_training
    self.reuse = False

  def __call__(self, input):
    with tf.variable_scope(self.name):
      C64 = convLayer(input, 64, reuse=self.reuse,
          is_training=self.is_training, name='C64')             # (?, w/2, h/2, 64)
      C128 = convLayer(C64, 128, reuse=self.reuse,
          is_training=self.is_training, name='C128')            # (?, w/4, h/4, 128)
      C256 = convLayer(C128, 256, reuse=self.reuse,
          is_training=self.is_training, name='C256')            # (?, w/8, h/8, 256)
      C512 = convLayer(C256, 512,reuse=self.reuse,
          is_training=self.is_training, name='C512')            # (?, w/16, h/16, 512)

      output = lastLayer(C512, reuse=self.reuse, name='output')          # (?, w/16, h/16, 1)

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output


def instance_norm(input):
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    # scale = _weights("scale", [depth], mean=1.0)
    scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02, dtype=tf.float32))
    # offset = _biases("offset", [depth])
    offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset