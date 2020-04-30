import tensorflow as tf
from reader import DataReader
from discriminator import Discriminator
from generator import Generator

REAL_LABEL = 0.9

class CycleGAN:
  def __init__(self):
   
    self.lambda1 = 10.0
    self.lambda2 = 10.0
    self.batch_size = 1
    self.lr = 2e-4
    self.beta1 = 0.5
    self.image_size = 256
    self.X_train_file = './data/tfrecords/apple.tfrecords'
    self.Y_train_file = './data/tfrecords/apple.tfrecords'

    self.is_training = tf.placeholder_with_default(True, shape=[], name='isTraining')

    self.G = Generator('Gen1', self.is_training, ngf=64, image_size=self.image_size)
    self.D_Y = Discriminator('Y_Dis', self.is_training)
    self.F = Generator('Gen2', self.is_training, image_size=self.image_size)
    self.D_X = Discriminator('X_Dis', self.is_training)

    fake_shape = [self.batch_size, self.image_size, self.image_size, 3]
    self.fake_x = tf.placeholder(tf.float32, shape=fake_shape)
    self.fake_y = tf.placeholder(tf.float32, shape=fake_shape)


  def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
    def make_optimizer(loss, variables, name='Adam'):
      global_step = tf.Variable(0, trainable=False)
      starter_lr = self.lr
      end_lr = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      lr = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_lr, global_step-start_decay_step,
                                            decay_steps, end_lr,
                                            power=1.0),
                  starter_lr
          )

      )
      tf.summary.scalar('lr/{}'.format(name), lr)

      learning_step = (
          tf.train.AdamOptimizer(lr, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step

    G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
    D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
    F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
    D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

    with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
      return tf.no_op(name='optimizers')

  def model(self):
    X_reader = DataReader(self.X_train_file, name='XReader', image_size=self.image_size, batch_size=self.batch_size)
    Y_reader = DataReader(self.Y_train_file, name='YReader', image_size=self.image_size, batch_size=self.batch_size)

    x = X_reader.read()
    y = Y_reader.read()

    cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)

    # X -> Y
    fake_y = self.G(x)
    G_gan_loss = self.generator_loss(self.D_Y, fake_y)
    G_loss =  G_gan_loss + cycle_loss
    D_Y_loss = self.discriminator_loss(self.D_Y, y, self.fake_y)

    # Y -> X
    fake_x = self.F(y)
    F_gan_loss = self.generator_loss(self.D_X, fake_x)
    F_loss = F_gan_loss + cycle_loss
    D_X_loss = self.discriminator_loss(self.D_X, x, self.fake_x)


    return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x

  

  def discriminator_loss(self, D, y, fake_y):
    error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL)) 
    error_fake = tf.reduce_mean(tf.square(D(fake_y)))

    loss = (error_real + error_fake) / 2
    return loss

  def generator_loss(self, D, fake_y):
    loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    return loss

  def cycle_consistency_loss(self, G, F, x, y):
    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
    loss = self.lambda1*forward_loss + self.lambda2*backward_loss
    return loss
