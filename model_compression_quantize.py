import tensorflow as tf
from tf_pose.networks import get_network

#######################################################################################
### $ python3 model_compression_quantize.py
#######################################################################################

def main():

    graph = tf.Graph()
    with graph.as_default():

        input_node = tf.placeholder(tf.float32, shape=(1, 368, 432, 3), name='image')
        net, pretrain_path, last_layer = get_network("mobilenet_v2_1.4", input_node, None, False)
        #net, pretrain_path, last_layer = get_network("mobilenet_v2_small", input_node, None, False)

        tf.contrib.quantize.create_eval_graph()

        saver = tf.train.Saver(tf.global_variables())
        sess  = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, 'models/train/test/model-30437')
        saver.save(sess, 'models/train/test/model-finalquant-30437')

        graphdef = graph.as_graph_def()
        tf.train.write_graph(graphdef, 'models/train/test', 'model-finalquant.pb', as_text=False)
        #tf.train.write_graph(graphdef, 'models/train/test', 'model-finalquant.pbtxt', as_text=True)

if __name__ == '__main__':
    main()
