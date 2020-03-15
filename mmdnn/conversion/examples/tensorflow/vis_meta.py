import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import os.path
import shutil
import sys
import argparse


def _get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ckpt',
        required=True,
        help='Path to the checkpoint meta file (.ckpt.meta).'
        )
    parser.add_argument(
        '--logdir',
        required=True,
        help='Path to the log directory for writing the graph summary for visualization.'
        )

    return parser


def visualize(ckpt, logdir):
    with tf.Session() as sess:
        tf.train.import_meta_graph(ckpt)
        train_writer = tf.summary.FileWriter(logdir)
        train_writer.add_graph(sess.graph)
        train_writer.close()


def _main():
    """
    Visualize the frozen TF graph using tensorboard.

    Arguments
    ----------
    --ckpt: path to the checkpoint meta file (.ckpt.meta)
    --logdir: path to the log directory for writing graph summary for visualization

    Usage
    ----------
    python vis_meta.py --ckpt=model.ckpt.meta --logdir=/tmp/pb


    To kill a previous tensorboard process, use the following commands in the terminal
    ps aux | grep tensorboard
    kill PID
    """

    parser = _get_parser()
    args, unknown_args = parser.parse_known_args()

    if not os.path.isfile(args.ckpt):
        print('The checkpoint meta file does not exist.')
        exit(1)

    if not os.path.isdir(args.logdir):
        print('The log directory does not exist.')
        exit(1)

    # Load file
    visualize(args.ckpt, args.logdir)

    # Run TensorBoard
    cmd = 'tensorboard --logdir={} {}'.format(
        args.logdir,
        ' '.join(unknown_args)
    )
    #print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    _main()
