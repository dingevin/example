import numpy as np
import tensorflow as tf


x = tf.placeholder(dtype=tf.float32, shape=[10, 512], name='input_x')
y = tf.placeholder(dtype=tf.float32, shape=[10, 1], name='input_y')
global_step = tf.get_variable(name='global_step', dtype=tf.int64, shape=[],
                              trainable=False, initializer=tf.zeros_initializer)

def dense(inputs, output_size, reuse=None, name=None):
  with tf.variable_scope(name, 'dense', reuse=reuse):
    if output_size == 128:
      input_size = 512
    else:
      input_size = 128
    w = tf.get_variable('kernel', [input_size, output_size])
    b = tf.get_variable('bias', [output_size], initializer=tf.zeros_initializer)
    outputs = tf.matmul(inputs, w) + b
    return outputs
    
   
with tf.device('/gpu:0'):

  max_seq_length = 10
  time = tf.constant(0, tf.int32)
  seq_out_states = tf.TensorArray(tf.float32, 10, name='attention_states')

  def _LoopContinue(time, seq_out_states):
    return time < max_seq_length

  def _LoopBody(time, seq_out_states):
    gather_x = tf.gather(x, [time])
    encoder_output = dense(gather_x, 128, name='dense')
    encoder_output.set_shape([1, 128])
    seq_out_states.write(time, encoder_output)
    return time+1, seq_out_states

  loop_vars = time, seq_out_states
  shape_invariants = tf.contrib.framework.nest.map_structure(
      lambda t: tf.TensorShape(None), loop_vars)

  (time, seq_out_states) = tf.while_loop(
    _LoopContinue,
    _LoopBody,
    loop_vars=loop_vars,
    shape_invariants=shape_invariants,
    parallel_iterations=1,
    swap_memory=False)
  encoder_output = seq_out_states.stack()
  encoder_output = tf.reshape(encoder_output, [-1, 128])

  with tf.variable_scope('output', reuse=None):
    logits = dense(encoder_output, 10, reuse=None, name='output')
    logits_softmax = tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(y*tf.log(logits_softmax))
    
  opt = tf.train.GradientDescentOptimizer(0.01)
  var_list = tf.trainable_variables()
  for var in var_list:
    print(var)
  grads_and_vars = opt.compute_gradients(cross_entropy)
  for gv in grads_and_vars:
    print(gv)
    
  grads_and_vars = average_gradients(grads_and_vars)
  train_step = opt.apply_gradients(grads_and_vars, global_step)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  while True:
    x_ = np.random.random((10, 512))
    y_ = np.zeros((10, 1))

    _, loss = sess.run([train_step, cross_entropy], feed_dict={x:x_, y:y_})
    print(loss)
