import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([3.0, 4.0], name="b")
result = a + b
print(a.graph is tf.get_default_graph())
