import tensorflow as tf

print(tf.__version__)

hell = tf.constant("hello world")
sess = tf.Session()
print(sess.run(hello))
