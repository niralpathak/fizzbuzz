import numpy as np
import tensorflow as tf

def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if   i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else:             return np.array([1, 0, 0, 0])

NUM_DIGITS = 12


trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 4096)])
trY = np.array([fizz_buzz_encode(i)          for i in range(101, 4096)])

NUM_HIDDEN = 200
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.03))

w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=py_x, logits=Y))
train_op = tf.train.GradientDescentOptimizer(0.2).minimize(cost)

predict_op = tf.argmax(py_x, 1)

def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

BATCH_SIZE = 250

actual = ['1','2','fizz','4','buzz','fizz','7','8','fizz','buzz','11','fizz','13','14','fizzbuzz','16','17','fizz','19','buzz','fizz','22','23','fizz','buzz','26','fizz','28','29','fizzbuzz','31','32','fizz','34','buzz','fizz','37','38','fizz','buzz','41','fizz','43','44','fizzbuzz','46','47','fizz','49','buzz','fizz','52','53','fizz','buzz','56','fizz','58','59','fizzbuzz','61','62','fizz','64','buzz','fizz','67','68','fizz','buzz','71','fizz','73','74','fizzbuzz','76','77','fizz','79','buzz','fizz','82','83','fizz','buzz','86','fizz','88','89','fizzbuzz','91','92','fizz','94','buzz','fizz','97','98','fizz','buzz']

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(2000):
        p = np.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        print(epoch, np.mean(np.argmax(trY, axis=1) ==
                             sess.run(predict_op, feed_dict={X: trX, Y: trY})))

    numbers = np.arange(1, 101)
    teX = np.transpose(binary_encode(numbers, NUM_DIGITS))

    teY = sess.run(predict_op, feed_dict={X: teX})
    output = np.vectorize(fizz_buzz)(numbers, teY)

    print(output)

correct = [(x == y) for x, y in zip(actual, output)]
print(sum(correct))