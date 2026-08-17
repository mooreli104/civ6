"""The TensorFlow policy/value network.

The legal-action set changes shape at every single decision (buy this card,
build that district on that hex, take this tech), so a fixed action-index
softmax does not fit the game. The network *scores candidates* instead:

    state   (B, S)     -> state trunk  -> h_s (B, H)
    actions (B, C, A)  -> action trunk -> h_a (B, C, H)
    logit(b, c) = MLP([h_s(b), h_a(b, c), h_s(b) * h_a(b, c)])

A masked softmax over the C candidate slots gives the policy; a second head
reads the state trunk and predicts the final (normalized) game score, which
regularizes the shared trunk and gives the evaluator something to inspect.

The helper layers are registered as serializable so a saved model reloads with
plain `keras.models.load_model(path)` - no custom_objects, no safe_mode juggling.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

NEG_INF = -1e9


@keras.saving.register_keras_serializable(package="civ6")
class BroadcastState(layers.Layer):
    """Repeat the state embedding once per candidate slot."""

    def call(self, inputs):
        state_emb, action_emb = inputs
        n = tf.shape(action_emb)[1]
        return tf.tile(tf.expand_dims(state_emb, 1), [1, n, 1])


@keras.saving.register_keras_serializable(package="civ6")
class MaskLogits(layers.Layer):
    """Push padded / illegal candidate slots to -inf before the softmax."""

    def call(self, inputs):
        logits, mask = inputs
        return logits + (1.0 - mask) * NEG_INF


@keras.saving.register_keras_serializable(package="civ6")
class SqueezeLast(layers.Layer):
    def call(self, x):
        return tf.squeeze(x, axis=-1)


def build_model(state_dim: int, action_dim: int, *, hidden: int = 256, embed: int = 128,
                dropout: float = 0.1, l2: float = 1e-6) -> keras.Model:
    reg = keras.regularizers.l2(l2)
    state_in = keras.Input(shape=(state_dim,), name="state", dtype="float32")
    actions_in = keras.Input(shape=(None, action_dim), name="actions", dtype="float32")
    mask_in = keras.Input(shape=(None,), name="mask", dtype="float32")

    # --- state trunk ---
    x = layers.LayerNormalization(name="state_norm")(state_in)
    x = layers.Dense(hidden, activation="relu", kernel_regularizer=reg, name="state_1")(x)
    x = layers.Dropout(dropout, name="state_drop")(x)
    x = layers.Dense(embed, activation="relu", kernel_regularizer=reg, name="state_2")(x)
    state_emb = layers.LayerNormalization(name="state_emb")(x)

    # --- action trunk (weights shared across candidates) ---
    a = layers.LayerNormalization(name="action_norm")(actions_in)
    a = layers.Dense(hidden, activation="relu", kernel_regularizer=reg, name="action_1")(a)
    a = layers.Dropout(dropout, name="action_drop")(a)
    a = layers.Dense(embed, activation="relu", kernel_regularizer=reg, name="action_2")(a)
    action_emb = layers.LayerNormalization(name="action_emb")(a)

    # --- state x action interaction ---
    tiled = BroadcastState(name="broadcast_state")([state_emb, action_emb])
    pair = layers.Concatenate(name="pair")(
        [tiled, action_emb, layers.Multiply(name="interaction")([tiled, action_emb])])
    h = layers.Dense(hidden, activation="relu", kernel_regularizer=reg, name="score_1")(pair)
    h = layers.Dropout(dropout, name="score_drop")(h)
    h = layers.Dense(embed // 2, activation="relu", kernel_regularizer=reg, name="score_2")(h)
    raw = layers.Dense(1, kernel_regularizer=reg, name="score_out")(h)
    raw = SqueezeLast(name="score_squeeze")(raw)
    logits = MaskLogits(name="logits")([raw, mask_in])

    # --- auxiliary value head: predicted final score, normalized ---
    v = layers.Dense(hidden // 2, activation="relu", kernel_regularizer=reg, name="value_1")(state_emb)
    value = layers.Dense(1, name="value_dense")(v)
    value = SqueezeLast(name="value")(value)

    return keras.Model(inputs={"state": state_in, "actions": actions_in, "mask": mask_in},
                       outputs={"logits": logits, "value": value}, name="civ6_policy")
