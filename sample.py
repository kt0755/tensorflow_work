import tensorflow as tf

print(tf.__version__)


### task 1
#hello = tf.constant("hello world")  # making a node
#sess = tf.Session()  # why need a session?? 
#print(sess.run(hello))

### task 2
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
print("node1:", node1, "node2:", node2)
print("node3:", node3)
sess = tf.Session()
print("session: node1, node2 ", sess.run([node1, node2]))
print("session: node3 ", sess.run(node3))
