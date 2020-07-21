import tensorflow as tf


def MultiStepLR(initial_learning_rate, lr_steps, lr_rate, name='MultiStepLR'):
    """Multi-steps learning rate scheduler."""
    lr_steps_value = [initial_learning_rate]
    for _ in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate)
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_steps, values=lr_steps_value)


def MultiStepWarmUpLR(initial_learning_rate, lr_steps, lr_rate,
                      warmup_steps=0., min_lr=0.,
                      name='MultiStepWarmUpLR'):
    """Multi-steps warm up learning rate scheduler."""
    assert warmup_steps <= lr_steps[0]
    assert min_lr <= initial_learning_rate
    lr_steps_value = [initial_learning_rate]
    for i in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate[i])
    return PiecewiseConstantWarmUpDecay(
        boundaries=lr_steps, values=lr_steps_value, warmup_steps=warmup_steps,
        min_lr=min_lr)

class PiecewiseConstantWarmUpDecay(
        tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule wiht warm up schedule.
    Modified from tf.keras.optimizers.schedules.PiecewiseConstantDecay"""

    def __init__(self, boundaries, values, warmup_steps, min_lr,
                 name=None):
        super(PiecewiseConstantWarmUpDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError(
                    "The length of boundaries should be 1 less than the"
                    "length of values")

        self.boundaries = boundaries
        self.values = values
        self.name = name
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def __call__(self, step):
        with tf.name_scope(self.name or "PiecewiseConstantWarmUp"):
            step = tf.cast(tf.convert_to_tensor(step), tf.float32)
            pred_fn_pairs = []
            warmup_steps = self.warmup_steps
            boundaries = self.boundaries
            values = self.values
            min_lr = self.min_lr

            pred_fn_pairs.append(
                (step <= warmup_steps,
                 # lambda: min_lr + step * (values[0] - min_lr) / warmup_steps))
                 lambda: tf.constant(values[0])))
            pred_fn_pairs.append(
                (tf.logical_and(step <= boundaries[0],
                                step > warmup_steps),
                 lambda: tf.constant(values[1])
                 # lambda : values[1] + values[2] - values[1]*(step - warmup_steps)/(boundaries[0] - warmup_steps)
                 ))
            pred_fn_pairs.append(
                (tf.logical_and(step <= boundaries[1],
                                step > boundaries[0]),
                 lambda: tf.constant(values[2])
                 # lambda: values[2] + values[3] - values[2]*(step - boundaries[0] ) / (boundaries[1] - boundaries[0] )
                 ))
            pred_fn_pairs.append(
                (tf.logical_and(step <= boundaries[2],
                                step > boundaries[1]),
                 lambda: tf.constant(values[3])
                 # lambda: values[3] + 1e-7 - values[3] * (step - boundaries[1] ) / (boundaries[2] - boundaries[1])
                 ))

            pred_fn_pairs.append(
                (step > boundaries[2], lambda: tf.constant(1e-5)))

            for low, high, v in zip(boundaries[:-1], boundaries[1:],
                                    values[1:-1]):
                # Need to bind v here; can do this with lambda v=v: ...
                pred = (step > low) & (step <= high)
                pred_fn_pairs.append((pred, lambda: tf.constant(v)))

            # The default isn't needed here because our conditions are mutually
            # exclusive and exhaustive, but tf.case requires it.
            return tf.case(pred_fn_pairs, lambda: tf.constant(values[0]),
                           exclusive=False)

    def get_config(self):
        return {
                "boundaries": self.boundaries,
                "values": self.values,
                "warmup_steps": self.warmup_steps,
                "min_lr": self.min_lr,
                "name": self.name
        }