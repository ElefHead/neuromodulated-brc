## Follows https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/layers/recurrent_v2.py#L68-L183

import tensorflow as tf
from tensorflow import keras 

import warnings

class NBRCell(keras.layers.GRUCell):
    def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=2,
               **kwargs):

        super(NBRCell, self).__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            reset_after=True,
            **kwargs
        )

    def call(self, inputs, states, training=None):
        ## GRU
        ## r_t = self.recurrent_activation(tf.matmul(U_r, x_t) + tf.matmul(W_r, h_tm1))
        ## z_t = self.recurrent_activation(tf.matmul(U_z, x_t) + tf.matmul(W_z, h_tm1))
        ## h_t = z_t * h_tm1 + (1 - z_t) * self.activation( tf.matmul(U_h, x_t) + r_t * tf.matmul(W_h, h_tm1) )

        ## Neuromodulated Bistable Recurrent Cell
        ## r_t = 1 + self.activation(tf.matmul(U_r, x_t) + tf.matmul(W_r, h_tm1))
        ## z_t = self.recurrent_activation(tf.matmul(U_z, x_t) + tf.matmul(W_z, h_tm1))
        ## h_t = (z_t * h_tm1) + (1 - z_t)*( self.activation( tf.matmul(U_h, x_t) + r_t * h_tm1 ))

        h_tm1 = states[0] if tf.nest.is_nested(states) else states  # previous memory

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=3)

        if self.use_bias:
            if not self.reset_after:
                input_bias, recurrent_bias = self.bias, None
            else:
                input_bias, recurrent_bias = tf.unstack(self.bias)

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs

            x_z = tf.matmul(inputs_z, self.kernel[:, :self.units])
            x_r = tf.matmul(inputs_r, self.kernel[:, self.units:self.units * 2])
            x_h = tf.matmul(inputs_h, self.kernel[:, self.units * 2:])

            if self.use_bias:
                x_z = tf.nn.bias_add(x_z, input_bias[:self.units])
                x_r = tf.nn.bias_add(x_r, input_bias[self.units: self.units * 2])
                x_h = tf.nn.bias_add(x_h, input_bias[self.units * 2:])

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1

            recurrent_z = tf.matmul(h_tm1_z, self.recurrent_kernel[:, :self.units])
            recurrent_r = tf.matmul(h_tm1_r,
                                self.recurrent_kernel[:, self.units:self.units * 2])
            if self.reset_after and self.use_bias:
                recurrent_z = tf.nn.bias_add(recurrent_z, recurrent_bias[:self.units])
                recurrent_r = tf.nn.bias_add(recurrent_r,
                                            recurrent_bias[self.units:self.units * 2])

            ## r_t = 1 + tanh(tf.matmul(U_r, x_t) + tf.matmul(W_r, h_tm1))
            ## z_t = self.recurrent_activation(tf.matmul(U_z, x_t) + tf.matmul(W_z, h_tm1))

            z = self.recurrent_activation(x_z + recurrent_z)
            r = 1 + tf.nn.tanh(x_r + recurrent_r)

            ## hh = self.activation( tf.matmul(U_h, x_t) + r_t * h_tm1 )) 
            recurrent_h = r *  h_tm1_h

            hh = self.activation( x_h + recurrent_h )

        else: 
            if 0. < self.dropout < 1.:
                inputs = inputs * dp_mask[0]

            # inputs projected by all gate matrices at once
            matrix_x = tf.matmul(inputs, self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                matrix_x = tf.nn.bias_add(matrix_x, input_bias)

            x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner = tf.matmul(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner = tf.nn.bias_add(matrix_inner, recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner = tf.matmul(h_tm1, self.recurrent_kernel[:, :2 * self.units])

            recurrent_z, recurrent_r, recurrent_h = tf.split(
                matrix_inner, [self.units, self.units, -1], axis=-1)

            ## r_t = 1 + tanh(tf.matmul(U_r, x_t) + tf.matmul(W_r, h_tm1))
            ## z_t = self.recurrent_activation(tf.matmul(U_z, x_t) + tf.matmul(W_z, h_tm1))

            r = 1 + tf.nn.tanh(x_r + recurrent_r)
            z = self.recurrent_activation(x_z + recurrent_z)

            ## hh =  self.activation( tf.matmul(U_h, x_t) + r_t * h_tm1 )) 
            recurrent_h = r * h_tm1

            hh = self.activation(x_h + recurrent_h)

        ## h_t = z_t * h_tm1 + (1 - z_t)*hh
        h = z * h_tm1 + (1 - z) * hh
        return h, [h]
