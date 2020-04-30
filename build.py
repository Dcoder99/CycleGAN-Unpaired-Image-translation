import tensorflow as tf
import random
import os

try:
  from os import scandir
except ImportError:
  from scandir import scandir


inputX = "data/apple2orange/trainA"
inputY = "data/apple2orange/trainB"

outputX = "data/tfrecords/apple.tfrecords"
outputY = "data/tfrecords/orange.tfrecords"

def data_reader(input_dir, shuffle=True):
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    shuffled_index = list(range(len(file_paths)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths


def _int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_path, image_buffer):
  file_name = file_path.split('/')[-1]
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
      'image/encoded_image': _bytes_feature((image_buffer))
    }))
  return example

def data_writer(input_dir, output_file):
  file_paths = data_reader(input_dir)
  output_dir = os.path.dirname(output_file)
  try:
    os.makedirs(output_dir)
  except os.error, e:
    pass
  images_num = len(file_paths)
  writer = tf.python_io.TFRecordWriter(output_file)

  for i in range(len(file_paths)):
    file_path = file_paths[i]

    with tf.gfile.FastGFile(file_path, 'rb') as f:
      image_data = f.read()

    example = _convert_to_example(file_path, image_data)
    writer.write(example.SerializeToString())

    if i % 500 == 0:
      print("Processed {}/{}.".format(i, images_num))
  print("Done.")
  writer.close()

def main(unused_argv):
  print("Converting X data to tfrecords...")
  data_writer(inputX, outputX)
  print("Converting Y data to tfrecords...")
  data_writer(inputY, outputY)

if __name__ == '__main__':
  tf.app.run()
