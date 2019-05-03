import os
import argparse
import tensorflow as tf
import graph_rewriter_builder
import exporter
from tf_pose.networks import get_network

#######################################################################################
### $ python3 model_compression.py
#######################################################################################

def main(quantize):

    graph = tf.Graph()
    with graph.as_default():

        input_node = tf.placeholder(tf.float32, shape=(1, 368, 432, 3), name='image')
        net, pretrain_path, last_layer = get_network("mobilenet_v2_1.4", input_node, None, False)

        if quantize == "True" or quantize == "true":
            graph_rewriter_fn = graph_rewriter_builder.build()
            graph_rewriter_fn()
            #exporter.rewrite_nn_resize_op(True)

            saver_kwargs = {}
            saver = tf.train.Saver(**saver_kwargs)
            input_saver_def = saver.as_saver_def()
            frozen_graph_def = exporter.freeze_graph_with_def_protos(
                input_graph_def=tf.get_default_graph().as_graph_def(),
                input_saver_def=input_saver_def,
                input_checkpoint='models/train/test/model_latest-2000',
                output_node_names='Openpose/concat_stage7',
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0',
                clear_devices=True,
                output_graph='',
                initializer_nodes='')

            transformed_graph_def = frozen_graph_def
            binary_graph = os.path.join("models/train/test", "tflite_graph.pb")
            with tf.gfile.GFile(binary_graph, 'wb') as f:
                f.write(transformed_graph_def.SerializeToString())

            txt_graph = os.path.join("models/train/test", "tflite_graph.pbtxt")
            with tf.gfile.GFile(txt_graph, 'w') as f:
                f.write(str(transformed_graph_def))

        else:
            saver = tf.train.Saver(tf.global_variables())
            sess  = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            saver.restore(sess, 'models/train/test/model_latest-114000') #<--- "-114000" changes according to the number of steps learned.
            saver.save(sess, 'models/train/test/model_latest-final-114000') #<--- "-114000" changes according to the number of steps learned.

            graphdef = graph.as_graph_def()
            tf.train.write_graph(graphdef, 'models/train/test', 'model_latest-final.pbtxt', as_text=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", default="False", help="Quantization disabled=False, Quantization enabled=True")
    args = parser.parse_args()
    main(args.quantize)
