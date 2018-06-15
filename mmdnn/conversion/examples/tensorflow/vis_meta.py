import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import shutil
import sys

def visualize(model_filename, log_dir):
    with tf.Session() as sess:
        tf.train.import_meta_graph(model_filename)
        train_writer = tf.summary.FileWriter(log_dir)
        train_writer.add_graph(sess.graph)
        train_writer.close()

def _main():
    """
    Visualize the frozen TF graph using tensorboard.

    Arguments
    ----------
    - path to the checkpoint meta file (.ckpt.meta)
    - path to a log directory for writing graph summary for visualization

    Usage
    ----------
    python vis_meta.py model.ckpt.meta /tmp/pb


    To kill a previous tensorboard process, use the following commands in the terminal
    ps aux | grep tensorboard
    kill PID
    """

    if len(sys.argv) != 3:
        raise ValueError("Usage: python vis_meta.py /path/to/model.meta /path/to/log/directory")
    # load file
    visualize(sys.argv[1], sys.argv[2])
    os.system("tensorboard --logdir=" + sys.argv[2])


if __name__ == "__main__":
    _main()