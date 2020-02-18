import tensorflow as tf
import numpy as np
from scipy.special import comb, beta

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

class MultiHeadAttention(tf.keras.layers.Attention):
    '''
    Refer1: DotProductAttention    
    Refer2: https://github.com/Kyubyong/transformer/blob/master/modules.py
    '''
    def __init__(self, num_heads, size, use_scale=False, **kwargs):
        super(MultiHeadAttention, self).__init__(use_scale= use_scale, **kwargs)

        if size % num_heads != 0:
            raise ValueError('size must be divisible by num_heads. (\'{}\' % \'{}\' != 0)'.format(size, num_heads))

        self.num_heads = num_heads
        self.size = size
        self.use_scale = use_scale

    def build(self, input_shape):
        self.layer_Dict = {
            'Query': tf.keras.layers.Dense(self.size),
            'Value': tf.keras.layers.Dense(self.size),
            'Key': tf.keras.layers.Dense(self.size),
            'Layer_Normalization': Layer_Norm()
            }

        super(MultiHeadAttention, self).build(input_shape= input_shape)

    def call(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = self.layer_Dict['Query'](inputs[0]) # [batch_size, Tq, Att_Dim]
        v = self.layer_Dict['Value'](inputs[1]) # [batch_size, Tv, Att_Dim]
        k = self.layer_Dict['Key'](inputs[2]) if len(inputs) > 2 else v # [batch_size, Tv, Att_Dim]

        #Multihead
        q_split = tf.concat(tf.split(q, self.num_heads, axis= -1), axis= 0)   # [batch_size * Heads, Tq, Att_Dim / Heads]
        v_split = tf.concat(tf.split(v, self.num_heads, axis= -1), axis= 0)   # [batch_size * Heads, Tv, Att_Dim / Heads]
        k_split = tf.concat(tf.split(k, self.num_heads, axis= -1), axis= 0)   # [batch_size * Heads, Tv, Att_Dim / Heads]
        
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None

        scores = self._calculate_scores(query= q_split, key= k_split)
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis= -2)
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
        result, attention_distribution = _apply_scores(scores=scores, value= v_split, scores_mask=scores_mask) #reslut: [batch_size * Heads, Tq, Att_Dim / Heads], attention_distribution: [batch_size * Heads, Tq, Tv]
        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = tf.expand_dims(q_mask, axis=-1)
            result *= tf.cast(q_mask, dtype=result.dtype)

        result = tf.concat(tf.split(result, self.num_heads, axis= 0), axis= -1)  # [batch_size, Tq, Att_Dim]

        result = self.layer_Dict['Layer_Normalization'](result + q) #Residual, layer normalization
        attention_distribution = tf.reduce_mean(tf.stack(tf.split(attention_distribution, self.num_heads, axis= 0), axis= 1), axis= 1)  # [batch_size * Heads, Tq, Tv] -> [batch_size, Heads, Tq, Tv] -> [batch_size, Tq, Tv]

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

def _apply_scores(scores, value, scores_mask=None):
    if scores_mask is not None:
        padding_mask = tf.logical_not(scores_mask)
        # Bias so padding positions do not contribute to attention distribution.
        scores -= 1.e9 * tf.cast(padding_mask, dtype= scores.dtype)
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

class Layer_Norm(tf.keras.layers.Layer):
    '''
    There are several restriction in 'tf.keras.layers.LayerNormalization'.    
    '''    
    def __init__(self, epsilon= 1e-8):
        super(Layer_Norm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.beta = self.add_weight(
            name= 'beta',
            shape= input_shape[-1:],
            initializer= tf.zeros_initializer(),
            dtype= self.dtype,
            trainable= True
            )
        self.gamma = self.add_weight(
            name= 'gamma',
            shape= input_shape[-1:],
            initializer= tf.ones_initializer(),
            dtype= self.dtype,
            trainable= True
            )

        self.built = True

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims= True)
        normalized = (inputs - mean) / ((variance + self.epsilon) ** .5)
        outputs = self.gamma * normalized + self.beta

        return outputs


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

    def call(self, inputs):
        '''
        inputs: [query, value] or [query, value, key]
        I don't implement the mask function now.
        '''
        self._validate_call_args(inputs=inputs, mask= None)
        query = self.layer_Dict['Query'](inputs[0])
        value = self.layer_Dict['Value'](inputs[1])
        key = self.layer_Dict['Key'](inputs[2]) if len(inputs) > 2 else value

        contexts = tf.zeros(shape= [tf.shape(query)[0], 1, self.size], dtype= query.dtype)  #initial attention, [Batch, 1, Att_dim]
        alignments = tf.zeros(shape= (tf.shape(query)[0], 1, tf.shape(key)[1]), dtype= query.dtype)   #initial alignment, [Batch, 1, T_k]

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

class BahdanauMonotonicAttention(tf.keras.layers.AdditiveAttention):
    '''
    Refer: https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/dense_attention.py#L307-L440
    This is for attention size managing and getting the attention history(scores).
    '''
    def __init__(
        self,
        size,
        sigmoid_noise= 0.0,
        normalize= False,
        **kwargs
        ):
        super(BahdanauMonotonicAttention, self).__init__(use_scale= False, **kwargs)
        
        self.size = size
        self.sigmoid_noise = sigmoid_noise
        self.normalize = normalize

    def build(self, input_shape):
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
        inputs: [query, value] or [query, value, key]
        I don't implement the mask function now.
        '''
        self._validate_call_args(inputs=inputs, mask= None)
        query = self.layer_Dict['Query'](inputs[0])
        value = self.layer_Dict['Value'](inputs[1])
        key = self.layer_Dict['Key'](inputs[2]) if len(inputs) > 2 else value

        contexts = tf.zeros(shape= [tf.shape(query)[0], 1, self.size], dtype= query.dtype)  #initial attention, [Batch, 1, Att_dim]
        alignments = tf.expand_dims(
            tf.one_hot(
                indices= tf.zeros((tf.shape(query)[0]), dtype= tf.int32),
                depth= tf.shape(key)[1],
                dtype= query.dtype
                ),
            axis= 1
            )   #initial alignment, [Batch, 1, T_k]. This part is different by monotonic or not.

        initial_Step = tf.constant(0)
        def body(step, query, contexts, alignments):
            query_Step = tf.expand_dims(query[:, step], axis= 1) #[Batch, 1, Att_dim]            
            previous_alignment = tf.expand_dims(alignments[:, -1], axis= 1) #[Batch, 1, T_k]

            score = self._calculate_scores(query= query_Step, key= key)   #[Batch, T_k]            
            context, alignment  = self._apply_scores(score= score, value= value, previous_alignment= previous_alignment) #[Batch, Att_dim], [Batch, T_v]

            return step + 1, query, tf.concat([contexts, context], axis= 1),  tf.concat([alignments, alignment], axis= 1)

        _, _, contexts, alignments = tf.while_loop(
            cond= lambda step, query, contexts, alignments: tf.less(step, tf.shape(query)[1]),
            body= body,
            loop_vars= [initial_Step, query, contexts, alignments],
            shape_invariants= [initial_Step.get_shape(), query.get_shape(), tf.TensorShape([None, None, self.size]), tf.TensorShape([None, None, None])]
            )

        return contexts[:, 1:], alignments[:, 1:]   #Remove initial step

    def _calculate_scores(self, query, key):
        """Calculates attention scores as a nonlinear sum of query and key.
        Args:
        query: Query tensor of shape `[batch_size, 1, Att_dim]`.
        key: Key tensor of shape `[batch_size, T_k, Att_dim]`.
        
        Returns:
        Tensor of shape `[batch_size, T_k]`.
        """
        if self.normalize:
            norm_v = self.attention_g * self.attention_v * tf.math.rsqrt(tf.reduce_sum(tf.square(self.attention_v)))
            return tf.reduce_sum(norm_v * tf.tanh(query + key + self.attention_b), axis= -1) + self.attention_score_bias   #[Batch, T_k, Att_dim] -> [Batch, T_k]
        else:
            return tf.reduce_sum(self.attention_v * tf.tanh(query + key), axis= -1) + self.attention_score_bias   #[Batch, T_k, Att_dim] -> [Batch, T_k]

    #In TF1, 'context' is calculated in AttentionWrapper, not attention mechanism.
    def _apply_scores(self, score, value, previous_alignment):
        '''
        score shape: [batch_size, T_v]`.    (Must T_k == T_v)
        value shape: [batch_size, T_v, Att_dim]`.
        previous_alignment shape: [batch_size, 1, T_v]`.
        

        Return: [batch_size, Att_dim]
        '''
        score = tf.expand_dims(score, axis= 1)  #[Batch_size, 1, T_v]        
        alignment = self._monotonic_probability_fn(score, previous_alignment)   #[Batch_size, 1, T_v]
        context = tf.matmul(alignment, value)   #[Batch_size, 1, Att_dim]
        
        return context, alignment

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

class StepwiseMonotonicAttention(BahdanauMonotonicAttention):
    '''
    Refer: https://gist.github.com/dy-octa/38a7638f75c21479582d7391490df37c
    '''
    def __init__(
        self,
        size,
        sigmoid_noise= 2.0,
        normalize= False,
        **kwargs
        ):
        super(StepwiseMonotonicAttention, self).__init__(
            size= size,
            sigmoid_noise= sigmoid_noise,
            normalize= normalize, **kwargs
            )

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


class DynamicConvolutionAttention(tf.keras.layers.AdditiveAttention):
    '''
    Refer: https://gist.github.com/attitudechunfeng/c162a5ed9b034be8f3f5800652af7c83
    '''
    def __init__(
        self,
        size,
        f_conv_filters= 8,
        f_conv_kernel_size= 21,
        f_conv_stride= 1,
        g_conv_filters= 8,
        g_conv_kernel_size= 21,
        g_conv_stride= [1, 1, 1, 1],
        p_conv_size = 11,
        p_alpha= 0.1,
        p_beta = 2.9,        
        use_scale=False,
        cumulate_weights= False,
        **kwargs
        ):
        super(DynamicConvolutionAttention, self).__init__(use_scale= use_scale, **kwargs)
        
        self.size = size
        self.f_conv_filters= f_conv_filters
        self.f_conv_kernel_size= f_conv_kernel_size
        self.f_conv_stride= f_conv_stride
        self.g_conv_filters= g_conv_filters
        self.g_conv_kernel_size= g_conv_kernel_size
        self.g_conv_stride= g_conv_stride
        self.p_conv_size = p_conv_size
        self.p_alpha= p_alpha
        self.p_beta = p_beta
        self.cumulate_weights = cumulate_weights
        
    def build(self, input_shape):
        self.layer_Dict = {}
        self.layer_Dict['Key'] = tf.keras.layers.Dense(self.size)

        self.layer_Dict['F_Conv'] = tf.keras.layers.Conv1D(
            filters= self.f_conv_filters,
            kernel_size= self.f_conv_kernel_size,
            strides= self.f_conv_stride,
            padding='same'
            )
        self.layer_Dict['F_Dense'] = tf.keras.layers.Dense(
            self.size,
            use_bias= False
            )
        
        self.layer_Dict['G_Filter_Dense'] = tf.keras.Sequential()
        self.layer_Dict['G_Filter_Dense'].add(tf.keras.layers.Dense(
            units= self.g_conv_kernel_size * self.g_conv_filters,
            use_bias= True,
            activation= 'tanh'
            ))
        self.layer_Dict['G_Filter_Dense'].add(tf.keras.layers.Dense(
            units= self.g_conv_kernel_size * self.g_conv_filters,
            use_bias= False
            ))
        self.layer_Dict['G_Dense'] = tf.keras.layers.Dense(
            self.size,
            use_bias= False
            )

        self.layer_Dict['P_Conv'] = DCA_P_Conv1D(
            p_conv_size = self.p_conv_size,
            p_alpha= self.p_alpha,
            p_beta = self.p_beta,
            )
        
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

        # self.g_scale = self.add_weight(
        #     name='g_scale',
        #     shape=[self.g_conv_kernel_size * self.g_conv_filters,],
        #     initializer=tf.zeros_initializer(),
        #     dtype=self.dtype,
        #     trainable=True
        #     )

        self.bulit = True

    def call(self, inputs):
        '''
        inputs: [query, key]
        I don't implement the mask function now.
        '''
        self._validate_call_args(inputs=inputs, mask= None)
        query = inputs[0]   #[Batch, Q_dim]
        key = self.layer_Dict['Key'](inputs[1]) #[Batch, T_k, Att_dim]

        batch_size = tf.shape(query)[0]
        contexts = tf.zeros(shape= [tf.shape(query)[0], 1, self.size], dtype= query.dtype)  #initial attention, [Batch, 1, Att_dim]
        alignments = tf.one_hot(
            indices= tf.zeros((tf.shape(query)[0], 1), dtype= tf.int32),
            depth= tf.shape(key)[1],
            dtype= query.dtype
            )   #initial alignment, [Batch, 1, T_k]. This part is different by monotonic or not.

        initial_Step = tf.constant(0)
        def body(step, query, contexts, alignments):
            query_Step = query[:, step] #[Batch, Q_dim]            
            previous_alignment = tf.reduce_sum(alignments, axis= 1) if self.cumulate_weights else alignments[:, -1] #[Batch, T_k]
            previous_alignment = tf.expand_dims(previous_alignment, axis= -1) #[Batch, T_k, 1]

            feature_previous_alignment = self.layer_Dict['F_Conv'](previous_alignment)    #[Batch, T_k, Filters]
            feature_previous_alignment = self.layer_Dict['F_Dense'](feature_previous_alignment)   #[Batch, T_k, Att_dim]

            #dynamic_filter = self.g_scale * self.layer_Dict['G_Filter_Dense'](query_Step)    # [Batch, Conv_Size * Conv_Ch]
            dynamic_filter = self.layer_Dict['G_Filter_Dense'](query_Step)    # [Batch, Conv_Size * Conv_Ch]
            dynamic_filter = tf.reshape(
                dynamic_filter,
                shape= [batch_size, 1, self.g_conv_kernel_size, self.g_conv_filters]
                )   # [Batch, 1, Conv_Size, Conv_Ch]
            dynamic_filter = tf.transpose(
                dynamic_filter,
                perm= [1, 2, 0, 3]
                )   # [1, Conv_Size, Batch, Conv_Ch]    [H(1), W, C_in, C_out]
            dynamic_previous_alignment = tf.expand_dims(
                tf.transpose(
                    previous_alignment,    
                    perm= [2, 1, 0]
                    ),   
                    axis = 0
                )   #[N(Batch), W(K_t), C(1)] -> [C(1), W(K_t), N(Batch)] -> [1, C(1), W(K_t), N(Batch)]
            dynamic_previous_alignment  = tf.nn.depthwise_conv2d(
                dynamic_previous_alignment,
                filter= dynamic_filter,
                strides= self.g_conv_stride,
                padding= 'SAME'
                )   # [1, 1, K_t, Batch * G_Filter]
            dynamic_previous_alignment = tf.squeeze(input= dynamic_previous_alignment, axis= [0, 1])  # [K_t, Batch * G_Filter]
            dynamic_previous_alignment = tf.reshape(
                dynamic_previous_alignment,
                shape= [tf.shape(dynamic_previous_alignment)[0], batch_size, self.g_conv_filters]
                )   # [K_t, Batch, G_Filter]
            dynamic_previous_alignment = tf.transpose(
                dynamic_previous_alignment,
                perm= [1, 0, 2]
                )   # [Batch, K_t, G_Filter]
            dynamic_previous_alignment = self.layer_Dict['G_Dense'](dynamic_previous_alignment)  #[Batch, K_t, Att_Dim]

            prior_filter_bias = self.layer_Dict['P_Conv'](previous_alignment)   #[Batch, K_t]

            score = self._calculate_scores(
                feature_previous_alignment= feature_previous_alignment,
                dynamic_previous_alignment= dynamic_previous_alignment,
                prior_filter_bias= prior_filter_bias
                )   #[Batch, T_k]
            context, alignment  = self._apply_scores(score= score, key= key) #[Batch, 1, Att_dim], [Batch, 1, T_k]
            
            return step + 1, query, tf.concat([contexts, context], axis= 1),  tf.concat([alignments, alignment], axis= 1)

        _, _, contexts, alignments = tf.while_loop(
            cond= lambda step, query, contexts, alignments: tf.less(step, tf.shape(query)[1]),
            body= body,
            loop_vars= [initial_Step, query, contexts, alignments],
            shape_invariants= [initial_Step.get_shape(), query.get_shape(), tf.TensorShape([None, None, self.size]), tf.TensorShape([None, None, None])]
            )   #[Batch, T_q + 1, Att_dim], [Batch, T_q + 1, T_k]

        return contexts[:, 1:], alignments[:, 1:]   #Remove initial step

    def _calculate_scores(self, feature_previous_alignment, dynamic_previous_alignment, prior_filter_bias):
        """Calculates attention scores as a nonlinear sum of query and key.
        Args:
        feature_previous_alignment: Location_features of shape `[batch_size, T_k, Att_dim]`.
        dynamic_previous_alignment: Dynamic features of shape `[batch_size, T_k, Att_dim]`.
        prior_filter_bias: Prior filter bias of shape `[batch_size, T_k]`.
        Returns:
        Tensor of shape `[batch_size, T_k]`.
        """
        if self.use_scale:
            scale = self.scale
        else:
            scale = 1.
        score = tf.reduce_sum(
            scale * tf.tanh(feature_previous_alignment + dynamic_previous_alignment + self.bias),
            axis=-1
            )   #[Batch, T_k, Att_dim] -> [Batch, T_k]
        return score + prior_filter_bias

    #In TF1, 'context' is calculated in AttentionWrapper, not attention mechanism.
    def _apply_scores(self, score, key):
        '''
        score shape: [batch_size, T_k]`.
        key shape: [batch_size, T_k, Att_dim]`.
        Must T_k == T_v

        Return: [batch_size, Att_dim]
        '''
        score = tf.expand_dims(score, axis= 1)  #[Batch_size, 1, T_v]
        alignment = tf.nn.softmax(score)   #[Batch_size, 1, T_v]
        context = tf.matmul(alignment, key)   #[Batch_size, 1, Att_dim]

        return context, alignment   #[Batch, 1, Att_dim], [Batch, 1, T_v]

class DCA_P_Conv1D(tf.keras.layers.Conv1D):
    def __init__(self, p_conv_size= 11, p_alpha= 0.1, p_beta= 0.9):
        self.p_conv_size= p_conv_size
        self.p_alpha= p_alpha
        self.p_beta= p_beta
        
        prior_filter = self.beta_binomial(self.p_conv_size, self.p_alpha, self.p_beta)
        prior_filter = np.flip(prior_filter, axis= 0)
        prior_filter = np.reshape(prior_filter, [self.p_conv_size, 1, 1])

        super(DCA_P_Conv1D, self).__init__(
            filters= 1,
            kernel_size= self.p_conv_size,
            padding='valid',
            use_bias= False,
            kernel_initializer= tf.initializers.constant(prior_filter)
            )
    
    def call(self, inputs):
        '''
        inputs: 3D tensor with shape: `(batch_size, steps, input_dim)`
        After front padding, call a superior class(Conv1D)
        '''
        inputs = tf.pad(inputs, paddings= [[0,0], [self.p_conv_size - 1, 0], [0, 0]])
        new_Tensor = super(DCA_P_Conv1D, self).call(inputs)
        new_Tensor = tf.squeeze(new_Tensor, axis= -1)
        
        return tf.math.log(tf.maximum(new_Tensor, np.finfo(inputs.dtype.as_numpy_dtype).tiny))
        # return tf.maximum(tf.math.log(new_Tensor), -1e+6) # NaN problem.

    def beta_binomial(self, _n, _alpha, _beta):   
        return [comb(_n,i) * beta(i+_alpha, _n-i+_beta) / beta(_alpha, _beta) for i in range(_n)]