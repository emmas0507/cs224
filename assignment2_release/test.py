import tensorflow as tf
import numpy as np

x = np.array(range(60))
x.shape = [12, 5]
label = np.array([[1,2,3],[2,3,4]])
x_ = tf.placeholder(tf.float32, [None, None])
label_ = tf.placeholder(tf.int32, [None, None])
embed = tf.nn.embedding_lookup(x_, label_)
sess = tf.Session()
sess.run(embed, feed_dict={x_: x, label_: label})

embed_list = []
for i in range(3):
    embed_list = embed_list + [embed[0, i]]

embed_reshape = tf.stack(embed_list, axis=1)
