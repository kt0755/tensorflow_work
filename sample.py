import tensorflow as tf

#print(tf.__version__)

### task 1
#hello = tf.constant("hello world")  # making a node ... building a graph??
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

### task 3 placeholder 

### taks 4 linear regression 
x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))  # cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # gradient descent 
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(cost), sess.run(W), sess.run(b))
