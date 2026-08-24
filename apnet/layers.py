import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="apnet")
class L2Normalize(Layer):
    def call(self, x):
        return tf.nn.l2_normalize(x, axis=1)

@register_keras_serializable(package="apnet")
class SNReLU(Layer):
    def call(self, x):
        return x * tf.nn.sigmoid(0.8 * x) + 0.1 * x / (1.0 + tf.abs(x))

@register_keras_serializable(package="apnet")
class ANASPReLU(Layer):
    def __init__(self, clip_value=3.0, **kwargs):
        super().__init__(**kwargs)
        self.clip_value = clip_value

    def build(self, input_shape):
        self.lmbda = self.add_weight(name="lambda", shape=(), initializer=tf.constant_initializer(-1.0), trainable=True)
        self.alpha = self.add_weight(name="alpha", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.beta = self.add_weight(name="beta", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.gamma = self.add_weight(name="gamma", shape=(), initializer=tf.constant_initializer(1.0), trainable=True)
        self.tau = self.add_weight(name="tau", shape=(), initializer=tf.constant_initializer(0.5), trainable=True)

    def call(self, x):
        lmbda = tf.nn.sigmoid(self.lmbda)       
        alpha = tf.nn.softplus(self.alpha) * 0.5  
        beta = tf.nn.softplus(self.beta) * 0.5   
        gamma = tf.clip_by_value(tf.nn.softplus(self.gamma), 1.0, 2.0)
        tau = tf.nn.softplus(self.tau) + 1e-6
        
        x_safe = tf.clip_by_value(x, -self.clip_value, self.clip_value)
        out_neg = lmbda * x_safe
        out_mid = alpha * tf.pow(tf.abs(x_safe), gamma) * tf.sign(x_safe)
        out_pos = alpha * tf.pow(tau, gamma) + beta * (x_safe - tau)
        return tf.where(x_safe <= 0, out_neg, tf.where(x_safe < tau, out_mid, out_pos))

    def get_config(self):
        config = super().get_config()
        config.update({"clip_value": self.clip_value})
        return config

@register_keras_serializable()
class AdaptiveLoss(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape=None):
        # Khai báo sẵn 4 biến log_var ngay khi khởi tạo mô hình
        self.log_vars = [
            self.add_weight(
                shape=(),
                initializer="zeros",
                trainable=True,
                name=f"log_var_{i}"
            )
            for i in range(4)
        ]
        super().build(input_shape)

    def call(self, losses, lambdas):
        total = 0.0
        for i, (L, lam) in enumerate(zip(losses, lambdas)):
            log_var = self.log_vars[i]
            total += (tf.exp(-log_var) * (lam * L) + log_var)
        return total