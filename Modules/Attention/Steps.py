import tensorflow as tf
import numpy as np

'''
TF 2.0's basic attention layers(Attention and AdditiveAttention) calculate parallelly.
TO USE MONOTONIC FUNCTION, ATTENTION MUST KNOW 'n-1 ALIGNMENT'.
Thus, this parallel versions do not support the monotonic function.
'''

class BahdanauAttention(tf.keras.layers.Layer):
    '''
    Refer: https://www.tensorflow.org/tutorials/text/nmt_with_attention
    '''
    def __init__(self, size):
        super(BahdanauAttention, self).__init__()
        self.size = size

    def build(self, input_shapes):
        self.layer_Dict = {
            'Query': tf.keras.layers.Dense(self.size),
            'Value': tf.keras.layers.Dense(self.size),
            'V': tf.keras.layers.Dense(1)
            }

        self.built = True

    def call(self, inputs):
        '''
        inputs: [queries, values]
        queries: [Batch, Query_dim]
        values: [Batch, T_v, Value_dim]
        '''
        queries, values = inputs

        queries = self.layer_Dict['Query'](queries) #[Batch, Att_dim]
        values = self.layer_Dict['Value'](values)   #[Batch, T_v, Att_dim]

        queries = tf.expand_dims(queries, 1)    #[Batch, 1, Att_dim]

        score = self.layer_Dict['V'](tf.nn.tanh(values + queries))  #[Batch, T_v, 1]

        attention_weights = tf.nn.softmax(score - tf.reduce_max(score, axis= 1, keepdims= True), axis=1)    #[Batch, T_v, 1]

        context_vector = tf.reduce_sum(attention_weights * values, axis=1)  #[Batch, T_v, Att_dim] -> [Batch, Att_dim]

        return context_vector, tf.squeeze(attention_weights, axis= -1)

    def initial_alignment_fn(self, batch_size, key_time, dtype):
        return tf.zeros((batch_size, key_time), dtype= dtype)

class BahdanauMonotonicAttention(tf.keras.layers.Layer):
    '''
    Refer
    https://www.tensorflow.org/tutorials/text/nmt_with_attention
    https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/seq2seq/attention_wrapper.py#L1004-L1175

    '''
    def __init__(self, size, sigmoid_noise= 0.0, normalize= False, **kwargs):
        super(BahdanauMonotonicAttention, self).__init__()

        self.size = size
        self.sigmoid_noise = sigmoid_noise
        self.normalize = normalize

    def build(self, input_shapes):
        self.layer_Dict = {
            'Query': tf.keras.layers.Dense(self.size),
            'Value': tf.keras.layers.Dense(self.size),
            'Key': tf.keras.layers.Dense(self.size)
            }

        self.attention_v = self.add_weight(
            name='attention_v',
            shape=[self.size,],
            initializer='glorot_uniform',
            dtype=self.dtype,
            trainable=True
            )

        self.attention_score_bias = self.add_weight(
            name='attention_score_bias',
            shape=[],
            initializer=tf.zeros_initializer(),
            dtype=self.dtype,
            trainable=True
            )

        if self.normalize:
            self.attention_g = self.add_weight(
                name='attention_g',
                shape=[],
                initializer= tf.initializers.constant([np.sqrt(1. / self.size),]),
                dtype=self.dtype,
                trainable=True
                )

            self.attention_b = self.add_weight(
                name='attention_b',
                shape=[self.size,],
                initializer= tf.zeros_initializer(),
                dtype=self.dtype,
                trainable=True
                )

        self.bulit = True

    def call(self, inputs):
        '''
        inputs: [queries, values, previous_alignments] or [queries, values, keys, previous_alignments]
        query: [Batch, Query_dim]
        value: [Batch, T_v, Value_dim]
        key: [Batch, T_v, Key_dim]
        previous_alignment: [Batch, T_v]
        '''
        if len(inputs) == 3:
            query, value, previous_alignment = inputs
        elif len(inputs) == 4:
            query, value, key, previous_alignment = inputs
        else:
            raise ValueError('Unexpected input length')
        
        query = self.layer_Dict['Query'](query) # [Batch, Att_dim]
        value = self.layer_Dict['Value'](value) # [Batch, T_v, Att_dim]
        key = self.layer_Dict['Key'](key) if len(inputs) == 4 else value   # [Batch, T_v, Att_dim]
        
        query = tf.expand_dims(query, 1)    # [Batch, 1, Att_dim]
        previous_alignment = tf.expand_dims(previous_alignment, axis= 1)  # [Batch, 1, T_v]

        score = self._calculate_scores(query= query, key= key)
        context, alignment  = self._apply_scores(
            score= score,
            value= value,
            previous_alignment= previous_alignment
            ) # [Batch, Att_dim], [Batch, 1, T_v]

        return context, alignment

    def _calculate_scores(self, query, key):
        '''
        Calculates attention scores as a nonlinear sum of query and key.
        Args:
        query: Query tensor of shape `[batch_size, 1, Att_dim]`.
        key: Key tensor of shape `[batch_size, T_k, Att_dim]`.
        
        Returns:
        Tensor of shape `[batch_size, T_k]`.
        '''
        if self.normalize:
            norm_v = self.attention_g * self.attention_v * tf.math.rsqrt(tf.reduce_sum(tf.square(self.attention_v)))
            return tf.reduce_sum(norm_v * tf.tanh(query + key + self.attention_b), axis= -1) + self.attention_score_bias   #[Batch, T_k, Att_dim] -> [Batch, T_k]
        else:
            return tf.reduce_sum(self.attention_v * tf.tanh(query + key), axis= -1) + self.attention_score_bias   #[Batch, T_k, Att_dim] -> [Batch, T_k]

    def _apply_scores(self, score, value, previous_alignment):
        '''
        score shape: [batch_size, T_v]`.    (Must T_k == T_v)
        value shape: [batch_size, T_v, Att_dim]`.
        previous_alignment shape: [batch_size, 1, T_v]`.
        
        Return: [batch_size, Att_dim], [batch_size, T_v]
        '''
        score = tf.expand_dims(score, axis= 1)  #[Batch_size, 1, T_v]        
        alignment = self._monotonic_probability_fn(score, previous_alignment)   #[Batch_size, 1, T_v]
        context = tf.matmul(alignment, value)   #[Batch_size, 1, Att_dim]
        
        return tf.squeeze(context, axis= 1), tf.squeeze(alignment, axis= 1)

    def _monotonic_probability_fn(self, score, previous_alignment):
        if self.sigmoid_noise > 0.0:
            score += self.sigmoid_noise * tf.random.normal(tf.shape(score), dtype= score.dtype)
        p_choose_i = tf.sigmoid(score)

        cumprod_1mp_choose_i = self.safe_cumprod(1 - p_choose_i, axis= 2, exclusive= True)

        alignment = p_choose_i * cumprod_1mp_choose_i * tf.cumsum(
            previous_alignment / tf.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.),
            axis= 2
            )

        return alignment

    # https://github.com/tensorflow/addons/blob/9e9031133c8362fedf40f2d05f00334b6f7a970b/tensorflow_addons/seq2seq/attention_wrapper.py#L810
    def safe_cumprod(self, x, *args, **kwargs):
        """Computes cumprod of x in logspace using cumsum to avoid underflow.
        The cumprod function and its gradient can result in numerical instabilities
        when its argument has very small and/or zero values.  As long as the
        argument is all positive, we can instead compute the cumulative product as
        exp(cumsum(log(x))).  This function can be called identically to
        tf.cumprod.
        Args:
        x: Tensor to take the cumulative product of.
        *args: Passed on to cumsum; these are identical to those in cumprod.
        **kwargs: Passed on to cumsum; these are identical to those in cumprod.
        Returns:
        Cumulative product of x.
        """
        x = tf.convert_to_tensor(x, name='x')
        tiny = np.finfo(x.dtype.as_numpy_dtype).tiny
        return tf.exp(tf.cumsum(tf.math.log(tf.clip_by_value(x, tiny, 1)), *args, **kwargs))

    def initial_alignment_fn(self, batch_size, key_time, dtype):
        return tf.one_hot(
            indices= tf.zeros((batch_size), dtype= tf.int32),
            depth= key_time,
            dtype= dtype
            )

class StepwiseMonotonicAttention(BahdanauMonotonicAttention):
    '''
    Refer: https://gist.github.com/dy-octa/38a7638f75c21479582d7391490df37c
    '''
    def __init__(self, size, sigmoid_noise= 2.0, normalize= False, **kwargs):
        super(StepwiseMonotonicAttention, self).__init__(size, sigmoid_noise, normalize, **kwargs)

    def _monotonic_probability_fn(self, score, previous_alignment):
        '''
        score:  [Batch_size, 1, T_v]
        previous_alignment: [batch_size, 1, T_v]
        '''
        if self.sigmoid_noise > 0.0:
            score += self.sigmoid_noise * tf.random.normal(tf.shape(score), dtype= score.dtype)
        p_choose_i = tf.sigmoid(score)  # [Batch_size, 1, T_v]

        pad = tf.zeros([tf.shape(p_choose_i)[0], 1, 1], dtype=p_choose_i.dtype)    # [Batch_size, 1, 1]

        alignment = previous_alignment * p_choose_i + tf.concat(
            [pad, previous_alignment[:, :, :-1] * (1.0 - p_choose_i[:, :, :-1])], axis= -1)

        return alignment