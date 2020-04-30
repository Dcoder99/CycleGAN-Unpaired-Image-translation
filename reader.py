import tensorflow as tf

class DataReader():
  def __init__(self, tfrecords, image_size, batch_size, name=''):
    self.tfrecords = tfrecords
    self.image_size = image_size
    self.min_queue_examples = 1000
    self.batch_size = batch_size
    self.num_threads = 8
    self.reader = tf.TFRecordReader()
    self.name = name

  def read(self):
    with tf.name_scope(self.name):
      fname_queue = tf.train.string_input_producer([self.tfrecords])
      reader = tf.TFRecordReader()

      _, serialized_example = self.reader.read(fname_queue)
      features = tf.parse_single_example(
          serialized_example,
          features={
            'image/file_name': tf.FixedLenFeature([], tf.string),
            'image/encoded_image': tf.FixedLenFeature([], tf.string),
          })

      image_buffer = features['image/encoded_image']
      image = self.preProcess(tf.image.decode_jpeg(image_buffer, channels=3))
      images = tf.train.shuffle_batch(
            [image], num_threads=self.num_threads, batch_size=self.batch_size, min_after_dequeue=self.min_queue_examples,
            capacity=self.min_queue_examples + 3*self.batch_size)

      tf.summary.image('_input', images)
    return images

  def preProcess(self, image):
    image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = (image/127.5) - 1.0 
    image.set_shape([self.image_size, self.image_size, 3])
    return image
