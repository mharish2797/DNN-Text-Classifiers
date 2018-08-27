import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn,rnn_cell
mnist = input_data.read_data_sets("/temp/data/",one_hot=True)


hm_epochs=10
classes=10
batch_size=128

chunk_size=28
n_chunks=28
rnn_size=128


#height * width
x=tf.placeholder('float',[None,n_chunks,chunk_size])
y=tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,classes])),
             'biases':tf.Variable(tf.random_normal([classes]))}
    #none,28,28
    x = tf.transpose(x, [1,0,2])
   #28,none,28 
    x = tf.reshape(x, [-1, chunk_size])
    #none,28
    x = tf.split(x, n_chunks, 0)
    #28
    gru_cell = rnn_cell.GRUCell(rnn_size)
    # GRU cell with rnn_size units in the hidden layer 
    outputs, states = rnn.static_rnn(gru_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    #learning rate=.001
    optimizer=tf.train.RMSPropOptimizer(0.001).minimize(cost)
    #RMS PROP
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)
