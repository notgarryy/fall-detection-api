import tensorflow as tf

class ExponentialSmoothing(tf.keras.layers.Layer):
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        # inputs: (B, T, F)
        x = inputs
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        F = tf.shape(x)[2]

        # TensorArray for smoothed frames
        ta = tf.TensorArray(dtype=x.dtype, size=T)

        # initial frame
        prev = x[:, 0, :]       # (B, F)
        ta = ta.write(0, prev)

        def body(t, prev, ta):
            curr = x[:, t, :]                           # (B, F)
            smoothed = self.alpha * curr + (1 - self.alpha) * prev
            ta = ta.write(t, smoothed)
            return t + 1, smoothed, ta

        def cond(t, prev, ta):
            return t < T

        _, _, ta = tf.while_loop(cond, body, [1, prev, ta])

        out = ta.stack()                                # (T, B, F)
        out = tf.transpose(out, [1, 0, 2])              # (B, T, F)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

class PositionalEncoding(tf.keras.layers.Layer):
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[-1]   # <- dynamic 8

        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, d_model, 2, dtype=tf.float32)
            * -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32))
        )

        angle_rads = position * div_term

        # sin/cos pair
        pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)

        # truncate if odd
        pos_encoding = pos_encoding[:, :d_model]

        pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # (1, 30, d_model)

        # return broadcast addition
        return inputs + pos_encoding
    
def merge_padding_and_attention_mask(inputs, padding_mask, attention_mask):
    input_mask = None

    # Keras embedding mask when mask_zero=True
    if hasattr(inputs, "_keras_mask") and inputs._keras_mask is not None:
        input_mask = tf.cast(inputs._keras_mask, tf.bool)

    # Priority: padding_mask > attention_mask > keras automatic mask
    final_mask = None

    if padding_mask is not None:
        padding_mask = tf.cast(padding_mask, tf.bool)
        final_mask = padding_mask[:, tf.newaxis, :]      # [B, 1, T]

    if attention_mask is not None:
        attention_mask = tf.cast(attention_mask, tf.bool)
        final_mask = attention_mask                       # [B, T, T]

    if final_mask is None and input_mask is not None:
        final_mask = input_mask[:, tf.newaxis, :]         # [B, 1, T]

    return final_mask

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0.0,
        activation="relu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = tf.keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first

        # (TF 2.15 supports masking)
        self.supports_masking = True

    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        key_dim = hidden_dim // self.num_heads

        # ---- Multi-head self-attention ----
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        self.att_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.att_dropout = tf.keras.layers.Dropout(self.dropout)

        # ---- Feedforward network ----
        self.ffn_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.ffn_dense1 = tf.keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.ffn_dense2 = tf.keras.layers.Dense(
            hidden_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self.ffn_dropout = tf.keras.layers.Dropout(self.dropout)

    def call(
        self,
        inputs,
        padding_mask=None,
        attention_mask=None,
        training=None,
        return_attention_scores=False,
    ):
        mask = merge_padding_and_attention_mask(inputs, padding_mask, attention_mask)

        # ------- Self-attention -------
        x = inputs
        residual = x

        if self.normalize_first:
            x = self.att_norm(x)

        att_output, att_scores = self.mha(
            query=x,
            value=x,
            attention_mask=mask,
            return_attention_scores=True,
            training=training,
        )

        x = self.att_dropout(att_output, training=training)
        x = x + residual

        if not self.normalize_first:
            x = self.att_norm(x)

        # ------- Feedforward -------
        residual = x

        if self.normalize_first:
            x = self.ffn_norm(x)

        x = self.ffn_dense1(x)
        x = self.ffn_dense2(x)
        x = self.ffn_dropout(x, training=training)
        x = x + residual

        if not self.normalize_first:
            x = self.ffn_norm(x)

        return (x, att_scores) if return_attention_scores else x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": tf.keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
                "normalize_first": self.normalize_first,
            }
        )
        return cfg
