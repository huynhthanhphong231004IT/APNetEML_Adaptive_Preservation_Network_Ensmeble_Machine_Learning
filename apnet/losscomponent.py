import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
@register_keras_serializable()
class Loss_Component(tf.keras.Model):
    def pairwise_dist(A, B):
        A = tf.cast(A, tf.float32)
        B = tf.cast(B, tf.float32)
        A = tf.nn.l2_normalize(A, axis=1)
        B = tf.nn.l2_normalize(B, axis=1)
        return 1.0 - tf.matmul(A, B, transpose_b=True)

    def intra_class_loss(f, y, p):
        f = tf.cast(f, tf.float32)
        p_y = tf.gather(p, y)
        loss = tf.reduce_sum(tf.square(f - p_y),axis=1)
        return tf.reduce_mean(loss)
    def margin_loss(f, y, p, margin=0.5):
        dist = Loss_Component.pairwise_dist(f, p)
        pos = tf.gather_nd(dist,tf.stack([tf.range(tf.shape(y)[0]), y], axis=1))
        pos = tf.expand_dims(pos, 1)
        loss = tf.maximum(0.0, margin + pos - dist)
        mask = 1.0 - tf.one_hot(y, depth=tf.shape(p)[0])
        loss = loss * mask
        return tf.reduce_mean(loss)
    def prototype_separation_loss(p, sigma=0.3):
        dist = Loss_Component.pairwise_dist(p, p)
        mask = 1.0 - tf.eye(tf.shape(p)[0])
        loss = tf.exp(-dist / (2 * sigma**2))
        loss = loss * mask
        return tf.reduce_mean(loss)
    def uncertainty_loss(f, p):
        dist = Loss_Component.pairwise_dist(f, p)
        u = tf.reduce_mean(dist, axis=1)
        u = tf.clip_by_value(u, 1e-6, 1.0)
        return tf.reduce_mean(u * tf.math.log(u))