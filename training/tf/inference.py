import tensorflow as tf
import parse
import numpy as np
import argparse

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    graph = load_graph(args.frozen_model_filename)

    pos, probs, winner = parse.generate_fake_pos()
    pos = np.reshape(pos, (-1, 120, 8, 8))
    print(pos.shape)
        
    # We access the input and output nodes 
    training = graph.get_tensor_by_name('import/input_training:0')
    x = graph.get_tensor_by_name('import/input_planes:0')
    y = graph.get_tensor_by_name('import/policy_head:0')
    z = graph.get_tensor_by_name('import/value_head:0')
        
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        y_out, z_out = sess.run([y, z], feed_dict={
            x: pos, training: False
        })
        print(y_out.shape)
        print(z_out.shape)
