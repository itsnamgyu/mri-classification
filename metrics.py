import tensorflow as tf


class SlicewiseAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_slices = self.add_weight(name='ts', initializer='zeros')
        self.correct_slices = self.add_weight(name='cs', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        total = tf.reduce_sum(y_true)  # one-hot
        y_valid = tf.cast(tf.reduce_sum(y_true, axis=2), tf.bool)

        y_true = tf.argmax(y_true, axis=2)
        y_pred = tf.argmax(y_pred, axis=2)
        correct = (y_true == y_pred) & y_valid
        correct = tf.reduce_sum(tf.cast(correct, tf.int64))
        correct = tf.cast(correct, tf.float32)

        self.total_slices.assign_add(total)
        self.correct_slices.assign_add(correct)

    def result(self):
        return self.correct_slices / self.total_slices

    def reset_states(self):
        self.total_slices.assign(0)
        self.correct_slices.assign(0)
