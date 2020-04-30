import tensorflow as tf

def convLayer1(input, k, reuse=False, activation='relu', is_training=True, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights_shape = [7, 7, input.get_shape()[3], k]
    W_var = tf.get_variable("W_var", weights_shape,
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))

    padInput = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padInput, W_var,
        strides=[1, 1, 1, 1], padding='VALID')
    normalized = instance_norm(conv)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output


def convLayer2(input, k, reuse=False, is_training=True, name=None):
  with tf.variable_scope(name, reuse=reuse):
    weights_shape = [3, 3, input.get_shape()[3], k]
    W_var = tf.get_variable("W_var", weights_shape,
                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))
    conv = tf.nn.conv2d(input, W_var,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = instance_norm(conv)
    output = tf.nn.relu(normalized)
    return output

def residualBlock(input, k,  reuse=False, is_training=True, name=None):
  with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('layer1', reuse=reuse):
      weights_shape = [3, 3, input.get_shape()[3], k]
      W_var1 = tf.get_variable("W_var1", weights_shape,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))

      padded_input = tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv1 = tf.nn.conv2d(padded_input, W_var1,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized1 = instance_norm(conv1)
      relu1 = tf.nn.relu(normalized1)

    with tf.variable_scope('layer2', reuse=reuse):
      weights_shape = [3, 3, relu1.get_shape()[3], k]
      W_var2 = tf.get_variable("W_var2", weights_shape,
                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))

      padded_relu = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv2 = tf.nn.conv2d(padded_relu, W_var2,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized2 = instance_norm(conv2)
    output = input+normalized2
    return output

def resNet(input, reuse, is_training=True, n=6):
  depth = input.get_shape()[3]
  for i in range(1,n+1):
    output = residualBlock(input, depth, reuse, is_training, 'R{}_{}'.format(depth, i))
    input = output
  return output


def deConvLayer(input, k, reuse=False, is_training=True, name=None, output_size=None):
  with tf.variable_scope(name, reuse=reuse):
    input_shape = input.get_shape().as_list()
    weights_shape = [3, 3, k, input_shape[3]]
    W_var = tf.get_variable("W_var", weights_shape,
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02, dtype=tf.float32))

    if not output_size:
      output_size = input_shape[1]*2
    output_shape = [input_shape[0], output_size, output_size, k]
    fsconv = tf.nn.conv2d_transpose(input, W_var,
        output_shape=output_shape,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = instance_norm(fsconv)
    output = tf.nn.relu(normalized)
    return output



class Generator:
  def __init__(self, name, is_training, ngf=64, image_size=128):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.is_training = is_training
    self.image_size = image_size

  def __call__(self, input):
    with tf.variable_scope(self.name):
      C64 = convLayer1(input, self.ngf, is_training=self.is_training,
          reuse=self.reuse, name='C64')                           
      C128 = convLayer2(C64, 2*self.ngf, is_training=self.is_training,
          reuse=self.reuse, name='C128')                              
      C256 = convLayer2(C128, 4*self.ngf, is_training=self.is_training,
          reuse=self.reuse, name='C256')                                
      res_output = resNet(C256, reuse=self.reuse, n=6)    

      D128 = deConvLayer(res_output, 2*self.ngf, is_training=self.is_training,
          reuse=self.reuse, name='D128')                                 
      D64 = deConvLayer(D128, self.ngf, is_training=self.is_training,
          reuse=self.reuse, name='D64', output_size=self.image_size)       


      output = convLayer1(D64, 3,
          activation='tanh', reuse=self.reuse, name='output')          

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output


def instance_norm(input):
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02, dtype=tf.float32))
    offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset