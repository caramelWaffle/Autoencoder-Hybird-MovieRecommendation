from functools import partial
import numpy as np
import tensorflow as tf

dataset = np.load("/Users/macintoshhd/thesis_recommendation/data/embedded_movie_content.npy")
BATCH_SIZE = 1
batch_size = tf.placeholder(tf.int64)
n_inputs = dataset.shape
x = tf.placeholder(tf.float32, shape=[None,n_inputs])

## Dataset
dataset = tf.data.Dataset.from_tensor_slices(x).repeat().batch(batch_size)
iter = dataset.make_initializable_iterator() # create the iterator
features = iter.get_next()

x = tf.placeholder(tf.float32, shape=[None,n_inputs])

## Encoder
n_hidden_1 = 300
n_hidden_2 = 150  # codings

## Decoder
n_hidden_3 = n_hidden_1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

## Define the Xavier initialization
xav_init =  tf.contrib.layers.xavier_initializer()
## Define the L2 regularizer
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)

## Create the dense layer
dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.relu,
                         kernel_initializer=xav_init,
                         kernel_regularizer=l2_regularizer)

## Make the mat mul
hidden_1 = dense_layer(features, n_hidden_1)
hidden_2 = dense_layer(hidden_1, n_hidden_2)
hidden_3 = dense_layer(hidden_2, n_hidden_3)
outputs = dense_layer(hidden_3, n_outputs, activation=None)

loss = tf.reduce_mean(tf.square(outputs - features))

## Optimize
loss = tf.reduce_mean(tf.square(outputs - features))
optimizer = tf.train.AdamOptimizer(learning_rate)
train  = optimizer.minimize(loss)


## Set params
n_epochs = 100

## Call Saver to save the model and re-use it later during evaluation
saver = tf.train.Saver()

n_batches = dataset.shape[0] / batch_size
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict={x: dataset,
                                          batch_size: BATCH_SIZE})
    print('Training...')
    print(sess.run(features).shape)
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            sess.run(train)
        if epoch % 10 == 0:
            loss_train = loss.eval()   # not shown
            print("\r{}".format(epoch), "Train MSE:", loss_train)
        #saver.save(sess, "./my_model_all_layers.ckpt")
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in path: %s" % save_path)