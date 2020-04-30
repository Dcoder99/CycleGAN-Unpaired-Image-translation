import os
import tensorflow as tf
from datetime import datetime
from model import CycleGAN
import random as rand

class ImagePool:
  def __init__(self, pool_size):
    self.poolSize = pool_size
    self.allImages = []

  def pick(self, image):
    if self.poolSize == 0:
      return image

    elif len(self.allImages) < self.poolSize:
      self.allImages.append(image)
      return image

    else:
      if rand.random() > 0.5:
        someNum = rand.randrange(0, self.pool_size)
        temp = self.allImages[someNum].copy()
        self.allImages[someNum] = image.copy()
        return temp
      else:
        return image


use_trained_model = None
imagePoolSize = 50

def train():
  if use_trained_model is not None:
    checkpoints_dir = "/content/gdrive/My Drive/checkpoints/" + use_trained_model
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "/content/gdrive/My Drive/checkpoints/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    cycle_gan = CycleGAN()
    G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = cycle_gan.model()
    allOptimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()

  with tf.Session(graph=graph) as session:
    if use_trained_model is not None:
      print("---Loading from prev checkpoint----")
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      graphMetaData = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(graphMetaData)
      restore.restore(session, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(graphMetaData.split("-")[2].split(".")[0])
      print("STEP--", step)
    else:
      session.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    try:
      fake_Y_pool = ImagePool(imagePoolSize)
      fake_X_pool = ImagePool(imagePoolSize)

      while not coord.should_stop():
        # get previously generated images
        fake_y_val, fake_x_val = session.run([fake_y, fake_x])

        # train
        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
              session.run(
                  [allOptimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                  feed_dict={cycle_gan.fake_y: fake_Y_pool.pick(fake_y_val),
                             cycle_gan.fake_x: fake_X_pool.pick(fake_x_val)}
              )
        )

        train_writer.add_summary(summary, step)
        train_writer.flush()

        if step % 1 == 0:
          print("Step:", step)
          print("G_loss:", G_loss_val)
          print("F_loss:", F_loss_val)
          print("D_X_loss:", D_X_loss_val)
          print("D_Y_loss:", D_Y_loss_val)

        if step % 20 == 0:
          save_path = saver.save(session, checkpoints_dir + "/model.ckpt", global_step=step)
          print("Model saved in file:", save_path)

        step += 1

    except KeyboardInterrupt:
      print("Keyboard Interrupt")
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(session, checkpoints_dir + "/model.ckpt", global_step=step)
      print("Model saved in file:", save_path)
      coord.request_stop()
      coord.join(threads)

def main(tmp_arg):
  train()

if __name__ == '__main__':
  tf.app.run()
