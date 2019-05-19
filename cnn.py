##### MNIST Classifier using CNN 

import tensorflow as tf

# download MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# define CNN model
def build_CNN_classifier(x):

  # MNIST data를 3 dimensional 형태로 reshape합니다. 
  # MNIST data는 grayscale img 기 때문에 3번째차원(컬러채널)의 값은 1입니다
  x_image = tf.reshape(x, [-1, 28, 28, 1])


  ###### 1st Conv Layer 
  # 32 Filters (5x5 Kernel Size)
  # 28x28x1 -> 28x28x32
  # relu( conv(W) + B )
  W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=5e-2))
  b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
  h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  # 1st Pooling Layer
  # Max Pooling을 이용해서 image 의 size 를 1/2로 downsample합니다.
  # 28x28x32 -> 14x14x32
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


  ###### 2nd Conv Layer 
  # 64 Filters (5x5 Kernel Size)
  # 14x14x32 -> 14x14x64
  W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2))
  b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

  # 2nd Pooling Layer
  # Max Pooling을 이용해서 image의 size를 1/2로 downsample합니다.
  # 14x14x64 -> 7x7x64
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


  ##### FC Layer
  # 7x7 size 를 가진 64개의 activation map을 1024개의 특징들로 변환합니다. 
  # 7x7x64(3136) -> 1024
  W_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=5e-2))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Output Layer ... softmax 
  # 1024개의 특징들(feature)을 10개의 class-one-hot encoding으로 표현된 숫자 0~9-로 변환합니다.
  # 1024 -> 10
  W_output = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=5e-2))
  b_output = tf.Variable(tf.constant(0.1, shape=[10]))
  logits = tf.matmul(h_fc1, W_output) + b_output
  y_pred = tf.nn.softmax(logits)

  return y_pred, logits


# input, output data 를 받기위한 placeholder 를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# declare CNN
y_pred, logits = build_CNN_classifier(x)

# Cross Entropy를 loss function으로 정의하고 옵티마이저를 정의합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# session 을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
  # initialize all variables 
  sess.run(tf.global_variables_initializer())

  # 10000 Step만큼 최적화를 수행합니다.
  for i in range(10000):
    # 50개씩 MNIST 데이터를 불러옵니다.
    batch = mnist.train.next_batch(50)
    # 100 Step마다 training 데이터셋에 대한 정확도를 출력합니다.
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
      print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f" % (i, train_accuracy))
    # 옵티마이저를 실행해 파라미터를 한스텝 업데이트합니다.
    sess.run([train_step], feed_dict={x: batch[0], y: batch[1]})

  # print test data accuracy 
  print("test data accuracy : %f" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))


