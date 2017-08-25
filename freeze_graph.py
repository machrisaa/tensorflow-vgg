# -*- coding: utf8 -*-
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import graph_util


def freeze_graph_as_pb(graph, out_node_names, sess=None, out_fname="graph.pb"):
    """Freeze Graph as Single Protobuf File

    params
    ======
    - graph: a `tf.Graph`, the graph to be freezed
    - sess: a `tf.Session`, the session object for retrieving parameters
    - out_fname (optional): string, the output pb file name (default: 'graph.pb')
    """
    if sess is None:
        sess = tf.Session(graph=graph)
    graph_def = graph.as_graph_def()
    with graph.as_default():
        init = tf.global_variables_initializer()

    with sess.as_default():
        init.run()
        freeze_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                     graph_def,
                                                                     out_node_names)
    with tf.gfile.GFile(out_fname, "wb") as fid:
        fid.write(freeze_graph_def.SerializeToString())


def import_pb_file(pb_file, **kwargs):
    """Import Protobuf File as Graph

    params
    ======
    - pb_file: string, the .pb file to be imported
    - kwargs: keyword arguments for `tf.import_graph_def`

    return
    ======
    A `tf.Graph` object
    """
    graph = tf.Graph()
    graph_def = graph.as_graph_def()
    with tf.gfile.GFile(pb_file, "rb") as fid:
        graph_def.ParseFromString(fid.read())

    with graph.as_default():
        tf.import_graph_def(graph_def, **kwargs)

    return graph


if __name__ == "__main__":
    """
    Demo script for how to freeze entire graph and load it back again
    (including architecture and parameters)
    """
    import argparse
    from vgg16 import Vgg16

    parser = argparse.ArgumentParser()
    parser.add_argument("vgg16_npy", nargs="?",
                        metavar="VGG16_NPY_PATH",
                        help="the vgg16 npy file path (default: vgg16.npy)",
                        default="vgg16.npy")
    parser.add_argument("-o", "--out-fname", dest="out_fname",
                        metavar="FILE.pb",
                        help="output file name (default: model.pb)",
                        default="model.pb")
    args = parser.parse_known_args()[0]

    # build and freeze the vgg16 graph
    vgg16 = Vgg16(args.vgg16_npy)
    vgg_graph = tf.Graph()
    with vgg_graph.as_default():
        rgb_images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="rgb_images")
        vgg16.build(rgb_images)
    freeze_graph_as_pb(vgg_graph, [vgg16.prob.op.name], out_fname=args.out_fname)
    print("vgg16 graph freezed...")

    # load it back again
    load_vgg16_graph = import_pb_file(args.out_fname, name="")
    print("vgg16 graph loaded...")
    ## do whatever you want here....

