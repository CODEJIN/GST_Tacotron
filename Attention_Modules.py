import tensorflow as tf
import numpy as np

'''
TF 2.0's basic attention layers(Attention and AdditiveAttention) calculate parallelly.
TO USE MONOTONIC FUNCTION, ATTENTION MUST KNOW 'n-1 ALIGNMENT'.
Thus, this parallel versions do not support the monotonic function.
'''

class DotProductAttention(tf.keras.layers.Attention):
    '''
    Refer: https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/dense_attention.py#L182-L303
    Changes
    1. Attention size managing
    2. Getting the attention history(scores).
    '''
    def __init__(self, size, use_scale=False, **kwargs):
        super(DotProductAttention, self).__init__(use_scale= use_scale, **kwargs)
        self.size = size
        self.layer_Dict = {
            'Query': tf.keras.layers.Dense(size),
            'Value': tf.keras.layers.Dense(size),
            'Key': tf.keras.layers.Dense(size)
            }

    def call(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = self.layer_Dict['Query'](inputs[0])
        v = self.layer_Dict['Value'](inputs[1])
        k = self.layer_Dict['Key'](inputs[2]) if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis=-2)
        if self.causal:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the future
            # into the past.
            scores_shape = tf.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]],
                axis=0)
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)
        result, attention_distribution = _apply_scores(scores=scores, value=v, scores_mask=scores_mask)
        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = tf.expand_dims(q_mask, axis=-1)
            result *= tf.cast(q_mask, dtype=result.dtype)

        return result, attention_distribution

    def _calculate_scores(self, query, key):
        """Calculates attention scores as a query-key dot product.
        Args:
        query: Query tensor of shape `[batch_size, Tq, dim]`.
        key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
        Tensor of shape `[batch_size, Tq, Tv]`.
        """
        scores = tf.matmul(query, key, transpose_b=True)

        if self.scale is not None:            
            scores *= self.scale
        return scores

class BahdanauAttention(tf.keras.layers.AdditiveAttention):
    '''
    Refer: https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/dense_attention.py#L307-L440
    This is for attention size managing and getting the attention history(scores).
    '''
    def __init__(self, size, use_scale=False, **kwargs):
        super(BahdanauAttention, self).__init__(use_scale= use_scale, **kwargs)
        self.size = size
        self.layer_Dict = {
            'Query': tf.keras.layers.Dense(size),
            'Value': tf.keras.layers.Dense(size),
            'Key': tf.keras.layers.Dense(size)
            }        

    def build(self, input_shape):
        if self.use_scale:
            self.scale = self.add_weight(
                name='scale',
                shape=[self.size],
                initializer= tf.initializers.glorot_uniform(),
                dtype=self.dtype,
                trainable=True)
        else:
            self.scale = None
        
        self.built = True

    def call(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = self.layer_Dict['Query'](inputs[0])
        v = self.layer_Dict['Value'](inputs[1])
        k = self.layer_Dict['Key'](inputs[2]) if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k) #[Batch, T_q, T_k]
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis=-2)
        if self.causal:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the future
            # into the past.
            scores_shape = tf.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]],
                axis=0)
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)

        result, attention_distribution = _apply_scores(scores=scores, value=v, scores_mask=scores_mask)
        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = tf.expand_dims(q_mask, axis=-1)
            result *= tf.cast(q_mask, dtype=result.dtype)
        
        return result, attention_distribution

    def _calculate_scores(self, query, key):
        """Calculates attention scores as a nonlinear sum of query and key.
        Args:
        query: Query tensor of shape `[batch_size, Tq, dim]`.
        key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
        Tensor of shape `[batch_size, Tq, Tv]`.
        """
        # Reshape tensors to enable broadcasting.
        # Reshape into [batch_size, Tq, 1, dim].
        q_reshaped = tf.expand_dims(query, axis=-2)
        # Reshape into [batch_size, 1, Tv, dim].
        k_reshaped = tf.expand_dims(key, axis=-3)
        if self.use_scale:
            scale = self.scale
        else:
            scale = 1.
        return tf.reduce_sum(
            scale * tf.tanh(q_reshaped + k_reshaped), axis=-1)

def _apply_scores(scores, value, scores_mask=None):
    if scores_mask is not None:
        padding_mask = tf.logical_not(scores_mask)
        # Bias so padding positions do not contribute to attention distribution.
        scores -= 1.e9 * tf.cast(padding_mask, dtype=tf.keras.backend.floatx())
    attention_distribution = tf.nn.softmax(scores)

    return tf.matmul(attention_distribution, value), attention_distribution

def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = tf.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-2)
    col_index = tf.cumsum(
        tf.ones(shape=shape, dtype=tf.int32), axis=-1)
    return tf.greater_equal(row_index, col_index)

def _merge_masks(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return tf.logical_and(x, y)







# Refer: https://github.com/begeekmyfriend/tacotron/blob/60d6932f510bf591acb25620290868900b5c0a41/models/attention.py
class LocationSensitiveAttention(tf.keras.layers.AdditiveAttention):
    '''
    Refer: https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/dense_attention.py#L307-L440
    This is for attention size managing and getting the attention history(scores).
    '''
    def __init__(
        self,
        size,
        conv_filters,
        conv_kernel_size,
        conv_stride,
        smoothing= False,
        use_scale=False,
        cumulate_weights= True,
        **kwargs
        ):
        super(LocationSensitiveAttention, self).__init__(use_scale= use_scale, **kwargs)
        
        self.size = size
        self.smoothing = smoothing
        self.cumulate_weights = cumulate_weights        
        self.layer_Dict = {
            'Query': tf.keras.layers.Dense(size),
            'Value': tf.keras.layers.Dense(size),
            'Key': tf.keras.layers.Dense(size),
            'Alignment_Conv': tf.keras.layers.Conv1D(
                filters= conv_filters,
                kernel_size= conv_kernel_size,
                strides= conv_stride,
                padding='same'
                ),
            'Alignment_Dense': tf.keras.layers.Dense(size)
            }

    def build(self, input_shape):
        """Creates scale and bias variable if use_scale==True."""
        if self.use_scale:
            self.scale = self.add_weight(
                name='scale',
                shape=[self.size],
                initializer= tf.initializers.glorot_uniform(),
                dtype=self.dtype,
                trainable=True)            
        else:
            self.scale = None

        self.bias = self.add_weight(
            name='bias',
            shape=[self.size,],
            initializer=tf.zeros_initializer(),
            dtype=self.dtype,
            trainable=True
            )

        self.bulit = True
        #super(LocationSensitiveAttention, self).build(input_shape)

    def call(self, inputs):
        '''
        inputs: [query, value] or [query, value, key]
        I don't implement the mask function now.
        '''
        self._validate_call_args(inputs=inputs, mask= None)
        query = self.layer_Dict['Query'](inputs[0])
        value = self.layer_Dict['Value'](inputs[1])
        key = self.layer_Dict['Key'](inputs[2]) if len(inputs) > 2 else value

        contexts = tf.zeros(shape= [tf.shape(query)[0], 1, self.size])  #initial alignment, [Batch, 1, Att_dim]
        alignments = tf.expand_dims(
            tf.one_hot(
                indices= tf.zeros((tf.shape(query)[0]), dtype= tf.int32),
                depth= tf.shape(key)[1],
                dtype= tf.float32
                ),
            axis= 1
            )   #initial alignment, [Batch, 1, T_k]

        initial_Step = tf.constant(0)
        def body(step, query, contexts, alignments):
            query_Step = tf.expand_dims(query[:, step], axis= 1) #[Batch, 1, Att_dim]            
            previous_alignment = tf.reduce_sum(alignments, axis= 1) if self.cumulate_weights else alignments[:, -1]
            location_features = tf.expand_dims(previous_alignment, axis= -1) #[Batch, T_k, 1]
            location_features = self.layer_Dict['Alignment_Conv'](location_features)    #[Batch, T_k, Filters]
            location_features = self.layer_Dict['Alignment_Dense'](location_features)   #[Batch, T_k, Att_dim]

            score = self._calculate_scores(query= query_Step, key= key, location_features= location_features)   #[Batch, T_k]
            context, alignment  = self._apply_scores(score= score, value= value) #[Batch, Att_dim], [Batch, T_v]

            return step + 1, query, tf.concat([contexts, context], axis= 1),  tf.concat([alignments, alignment], axis= 1)

        _, _, contexts, alignments = tf.while_loop(
            cond= lambda step, query, contexts, alignments: tf.less(step, tf.shape(query)[1]),
            body= body,
            loop_vars= [initial_Step, query, contexts, alignments],
            shape_invariants= [initial_Step.get_shape(), query.get_shape(), tf.TensorShape([None, None, self.size]), tf.TensorShape([None, None, None])]
            )

        # # The following code cannot use now because normal for-loop does not support 'shape_invariants'.
        # for step in tf.range(tf.shape(query)[1]):
        #     query_Step = tf.expand_dims(query[:, step], axis= 1) #[Batch, 1, Att_dim]
        #     location_features = tf.expand_dims(alignments[:, -1], axis= -1) #[Batch, T_k, 1]
        #     location_features = self.layer_Dict['Alignment_Conv'](location_features)    #[Batch, T_k, Filters]
        #     location_features = self.layer_Dict['Alignment_Dense'](location_features)   #[Batch, T_k, Att_dim]

        #     score = self._calculate_scores(query= query_Step, key= key, location_features= location_features)   #[Batch, T_k]
        #     context, alignment  = self._apply_scores(score= score, value= value) #[Batch, Att_dim], [Batch, T_v]

        #     contexts = tf.concat([contexts, context], axis= 1)
        #     alignments = tf.concat([alignments, alignment], axis= 1)

        return contexts[:, 1:], alignments[:, 1:]   #Remove initial step

    def _calculate_scores(self, query, key, location_features):
        """Calculates attention scores as a nonlinear sum of query and key.
        Args:
        query: Query tensor of shape `[batch_size, 1, Att_dim]`.
        key: Key tensor of shape `[batch_size, T_k, Att_dim]`.
        location_features: Location_features of shape `[batch_size, T_k, Att_dim]`.
        Returns:
        Tensor of shape `[batch_size, T_k]`.
        """
        if self.use_scale:
            scale = self.scale
        else:
            scale = 1.

        return tf.reduce_sum(scale * tf.tanh(query + key + location_features + self.bias), axis=-1)    #[Batch, T_k, Att_dim] -> [Batch, T_k]

    #In TF1, 'context' is calculated in AttentionWrapper, not attention mechanism.
    def _apply_scores(self, score, value):
        '''
        score shape: [batch_size, T_k]`.
        value shape: [batch_size, T_v, Att_dim]`.
        Must T_k == T_v

        Return: [batch_size, Att_dim]
        '''
        score = tf.expand_dims(score, axis= 1)  #[Batch_size, 1, T_v]
        probability_fn = self._smoothing_normalization if self.smoothing else tf.nn.softmax
        alignment = probability_fn(score)   #[Batch_size, 1, T_v]
        context = tf.matmul(alignment, value)   #[Batch_size, 1, Att_dim]

        #return tf.squeeze(context, axis= 1), tf.squeeze(alignment, axis= 1),   #[Batch, Att_dim], [Batch, T_v]
        return context, alignment

    def _smoothing_normalization(self, e):
        """Applies a smoothing normalization function instead of softmax
        Introduced in:
            J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
        gio, “Attention-based models for speech recognition,” in Ad-
        vances in Neural Information Processing Systems, 2015, pp.
        577–585.
        ############################################################################
                            Smoothing normalization function
                    a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
        ############################################################################
        Args:
            e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
                values of an attention mechanism
        Returns:
            matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
                attendance to multiple memory time steps.
        """
        return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)