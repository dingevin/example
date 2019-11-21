import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops


def dense(inputs,
          output_size,
          activation=tf.identity,
          use_bias=True,
          reuse_kernel=None,
          reuse=None,
          name=None):
  with tf.variable_scope(name, "dense", reuse=reuse):
    input_size = inputs.get_shape().as_list()[-1]
    inputs_shape = tf.unstack(tf.shape(inputs))
    inputs = tf.reshape(inputs, [-1, input_size])
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_kernel):
      w = tf.get_variable("kernel", [output_size, input_size])
      outputs = tf.matmul(inputs, w, transpose_b=True)
      if use_bias:
        b = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer)
        outputs += b
        outputs = activation(outputs)
        return tf.reshape(outputs, inputs_shape[:-1] + [output_size])



class Model(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
          self.pls_batch_x = tf.placeholder(dtype=tf.float32, shape=[None, None, 512],
                                            name='src_pl')  # [batch_size, feat_len, feat_dim]
          self.pls_batch_y = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                            name='dst_pl')  # [batch_size, feat_len] 
        self._ff_activation = tf.nn.relu


    def prepare_training(self):
        with self.graph.as_default():
            # Optimizer
            self.global_step = tf.get_variable(name='global_step', dtype=tf.int64, shape=[],
                                              trainable=False, initializer=tf.zeros_initializer)

            self.learning_rate = tf.convert_to_tensor(0.01, dtype=tf.float32)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self._initializer = init_ops.variance_scaling_initializer(scale=1.0, mode='fan_avg', distribution='uniform')


    def build_train_model(self, reuse=None):
        """Build model for training. """
        tf.logging.info('Build train model.')
        self.prepare_training()

        with self.graph.as_default():
          acc_list, loss_list, gv_list = [], [], []
          with tf.variable_scope(tf.get_variable_scope(),
                                  initializer=self._initializer,
                                  reuse=reuse):
            with tf.device('/gpu:0'):
              tf.logging.info('Build model on %s.' % '/gpu:0')

              output_logits = self.forward(self.pls_batch_x, is_training=True, reuse=None,
                                          scope='forward')
              acc, loss = self.train_output(output_logits, self.pls_batch_y, reuse=None,
                                          scope='output')
              var_list = tf.trainable_variables()
              for var in var_list:
                print(var)
              gv_list.append(self._optimizer.compute_gradients(loss, var_list=var_list))
              for gv in gv_list:
                print(gv)

          self.loss = loss
          # Clip gradients and then apply.
          grads_and_vars = self.average_gradients(gv_list)
          avg_abs_grads = tf.reduce_mean(tf.abs(grads_and_vars[0]))

          grads, self.grads_norm = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars],
                                                          clip_norm=5)
          grads_and_vars = zip(grads, [gv[1] for gv in grads_and_vars])
          self.train_op = self._optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)


    def average_gradients(self, tower_grads):
      average_grads = []
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        #print('-----------')
        #print(grad_and_vars)
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)
        else:
          # Average over the 'tower' dimension.
          grad = tf.concat(axis=0, values=grads)
          grad = tf.reduce_mean(grad, 0)

          # Keep in mind that the Variables are redundant because they are shared
          # across towers. So .. we will just return the first tower's pointer to
          # the Variable.
          v = grad_and_vars[0][1]
          grad_and_var = (grad, v)
          average_grads.append(grad_and_var)
      return average_grads

    def forward(self, input_X, is_training, reuse, scope):

        with tf.variable_scope(scope, reuse=reuse):
          max_seq_length = tf.shape(input_X)[1]-20
          # Initialize all loop variables.
          time = tf.constant(0, tf.int32)
          batch_size = tf.shape(input_X)[0]
          time_steps = tf.shape(input_X)[1]
          seq_out_states = tf.TensorArray(tf.float32, max_seq_length, name='attention_states')
          attention_state_zero = tf.zeros([batch_size, 21*512])

          def _LoopContinue(time, attention_state, seq_out_states):
            del attention_state, seq_out_states
            return time < max_seq_length

          def _LoopBody(time, old_attention_state, seq_out_states):
            per_step_padding = tf.pad(
              tf.ones([batch_size, 21]),
              [[0, 0], [time, time_steps-time-21]], mode='CONSTANT')
            encoder_output = tf.boolean_mask(input_X, per_step_padding)
            encoder_output = tf.reshape(encoder_output,
              (batch_size, 21, encoder_output.get_shape().as_list()[-1]))
            encoder_output = dense(encoder_output, 1024, name='dense')
            attention_state = tf.reshape(encoder_output, [batch_size, 21*1024])
            seq_out_states.write(time, attention_state)
            return time+1, attention_state, seq_out_states

          loop_vars = time, attention_state_zero, seq_out_states
          shape_invariants = tf.contrib.framework.nest.map_structure(
            lambda t: tf.TensorShape(None), loop_vars)

          (time, attention_state, seq_out_states) = tf.while_loop(
              _LoopContinue,
              _LoopBody,
              loop_vars=loop_vars,
              shape_invariants=shape_invariants,
              parallel_iterations=12,
              swap_memory=False)
          encoder_output = seq_out_states.stack()
          encoder_output = tf.transpose(encoder_output, (1, 0, 2))
          return encoder_output


    def train_output(self, output, true_Y, reuse, scope):
        """Calculate loss and accuracy.
        Args:
            output: A Tensor with shape [batch, time, num_classes]
            true_Y: A Tensor with shape [batch, time]
        """
        with tf.variable_scope(scope, reuse=reuse):
            logits = dense(output, 10, use_bias=True,
                            name='dense')

            mask = tf.to_float(tf.not_equal(true_Y, -1))
            labels = tf.one_hot(true_Y, 10, 1.0, 0.0) # -1 in true_Y 
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits) # [batch, time]
            loss = tf.multiply(loss, tf.cast(mask, tf.float32))                        # 
            loss_mean = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(mask, tf.float32))

            correct_preds = tf.cast(
                tf.equal(tf.argmax(tf.nn.softmax(logits), 2, output_type=tf.int32), true_Y), tf.float32)
            correct_next_preds = tf.reduce_sum(tf.multiply(correct_preds, tf.cast(mask, tf.float32)))
            acc = correct_next_preds / tf.reduce_sum(tf.cast(mask, tf.float32))

        return acc, loss_mean



def main(_):

  model = Model()
  model.build_train_model()

  with tf.Session() as sess:

    x = tf.truncated_normal(shape=[2, 50, 512], mean=0.0, stddev=1.0, dtype=tf.float32)
    y = tf.ones(shape=[2, 50], dtype=tf.int32)


    loss, _ = sess.run([model.loss, model.train_op], feed_dict={model.pls_batch_x:x, model.pls_batch_y:y})
    print(loss)




if __name__ == '__main__':
  tf.app.run(main)


