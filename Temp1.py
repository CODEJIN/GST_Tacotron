class BaseDenseAttention(Layer):
  """Base Attention class for Dense networks.
  This class is suitable for Dense or CNN networks, and not for RNN networks.
  Implementations of attention mechanisms should inherit from this class, and
  reuse the `apply_attention_scores()` method.
  Args:
    causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
      that position `i` cannot attend to positions `j > i`. This prevents the
      flow of information from the future towards the past.
  Call Arguments:
    inputs: List of the following tensors:
      * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
      * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
      * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
        given, will use `value` for both `key` and `value`, which is the
        most common case.
    mask: List of the following tensors:
      * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.
  Output shape:
    Attention outputs of shape `[batch_size, Tq, dim]`.
  """

  def __init__(self, causal=False, **kwargs):
    super(BaseDenseAttention, self).__init__(**kwargs)
    self.causal = causal
    self.supports_masking = True

  def _calculate_scores(self, query, key):
    """Calculates attention scores.
    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.
    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
    return NotImplementedError

  def _apply_scores(self, scores, value, scores_mask=None):
    """Applies attention scores to the given value tensor.
    To use this method in your attention layer, follow the steps:
    * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
      `[batch_size, Tv]` to calculate the attention `scores`.
    * Pass `scores` and `value` tensors to this method. The method applies
      `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
      returns `matmul(attention_distribution, value).
    * Apply `query_mask` and return the result.
    Args:
      scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
      value: Value tensor of shape `[batch_size, Tv, dim]`.
      scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
        `[batch_size, Tq, Tv]`. If given, scores at positions where
        `scores_mask==False` do not contribute to the result. It must contain
        at least one `True` value in each line along the last dimension.
    Returns:
      Tensor of shape `[batch_size, Tq, dim]`.
    """
    if scores_mask is not None:
      padding_mask = math_ops.logical_not(scores_mask)
      # Bias so padding positions do not contribute to attention distribution.
      scores -= 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx())
    attention_distribution = nn.softmax(scores)
    return math_ops.matmul(attention_distribution, value)

  # TODO(b/125916026): Consider exposing a __call__ method with named args.
  def call(self, inputs, mask=None):
    self._validate_call_args(inputs=inputs, mask=mask)
    q = inputs[0]
    v = inputs[1]
    k = inputs[2] if len(inputs) > 2 else v
    q_mask = mask[0] if mask else None
    v_mask = mask[1] if mask else None
    scores = self._calculate_scores(query=q, key=k)
    if v_mask is not None:
      # Mask of shape [batch_size, 1, Tv].
      v_mask = array_ops.expand_dims(v_mask, axis=-2)
    if self.causal:
      # Creates a lower triangular mask, so position i cannot attend to
      # positions j>i. This prevents the flow of information from the future
      # into the past.
      scores_shape = array_ops.shape(scores)
      # causal_mask_shape = [1, Tq, Tv].
      causal_mask_shape = array_ops.concat(
          [array_ops.ones_like(scores_shape[:-2]), scores_shape[-2:]],
          axis=0)
      causal_mask = _lower_triangular_mask(causal_mask_shape)
    else:
      causal_mask = None
    scores_mask = _merge_masks(v_mask, causal_mask)
    result = self._apply_scores(scores=scores, value=v, scores_mask=scores_mask)
    if q_mask is not None:
      # Mask of shape [batch_size, Tq, 1].
      q_mask = array_ops.expand_dims(q_mask, axis=-1)
      result *= math_ops.cast(q_mask, dtype=result.dtype)
    return result

  def compute_mask(self, inputs, mask=None):
    self._validate_call_args(inputs=inputs, mask=mask)
    if mask:
      q_mask = mask[0]
      if q_mask is None:
        return None
      return ops.convert_to_tensor(q_mask)
    return None

  def _validate_call_args(self, inputs, mask):
    """Validates arguments of the call method."""
    class_name = self.__class__.__name__
    if not isinstance(inputs, list):
      raise ValueError(
          '{} layer must be called on a list of inputs, namely [query, value] '
          'or [query, value, key].'.format(class_name))
    if len(inputs) < 2 or len(inputs) > 3:
      raise ValueError(
          '{} layer accepts inputs list of length 2 or 3, '
          'namely [query, value] or [query, value, key]. '
          'Given length: {}'.format(class_name, len(inputs)))
    if mask:
      if not isinstance(mask, list):
        raise ValueError(
            '{} layer mask must be a list, '
            'namely [query_mask, value_mask].'.format(class_name))
      if len(mask) != 2:
        raise ValueError(
            '{} layer mask must be a list of length 2, namely [query_mask, '
            'value_mask]. Given length: {}'.format(class_name, len(mask)))