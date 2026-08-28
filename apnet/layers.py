import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
from .losscomponent import Loss_Component
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
        self.log_vars = [
            self.add_weight(shape=(),initializer="zeros",trainable=True,name=f"log_var_{i}")
            for i in range(5)
            ]
        super().build(input_shape)

    def call(self, losses, lambdas):
        total = 0.0
        for i, (L, lam) in enumerate(zip(losses, lambdas)):
            log_var = self.log_vars[i]
            total += (tf.exp(-log_var) * (lam * L) + log_var)
        return total


@register_keras_serializable()
class PrototypeModel(tf.keras.Model):
    def __init__(self, feature_extractor, num_classes, embedding_dim):
        super().__init__()
        self.num_classes = num_classes
        self.stage = self.add_weight(name="stage",shape=(),dtype=tf.int32,initializer="zeros",trainable=False)
        self.encoder = feature_extractor
        self.prototypes = self.add_weight(
            shape=(num_classes, embedding_dim),
            initializer="glorot_normal",
            trainable=True,
            name="prototypes"
        )

        self.adaptive_loss = AdaptiveLoss()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")
        self.lce_tracker = tf.keras.metrics.Mean(name="Lce")
        self.liccl_tracker = tf.keras.metrics.Mean(name="Liccl")
        self.licmrl_tracker = tf.keras.metrics.Mean(name="Licmrl")
        self.lpsl_tracker = tf.keras.metrics.Mean(name="Lpsl")
        self.luaer_tracker = tf.keras.metrics.Mean(name="Luaer")

        self.base_lambda1 = 1.0
        self.base_lambda2 = 0.5
        self.base_lambda3 = 0.3
        self.base_lambda4 = 0.05

        self.lambda1 = self.add_weight(
            name="lambda1",
            shape=(),
            initializer=tf.constant_initializer(0.0),
            trainable=True
        )

        self.lambda2 = self.add_weight(
            name="lambda2",
            shape=(),
            initializer=tf.constant_initializer(0.0),
            trainable=True
        )

        self.lambda3 = self.add_weight(
            name="lambda3",
            shape=(),
            initializer=tf.constant_initializer(0.0),
            trainable=True
        )

        self.lambda4 = self.add_weight(
            name="lambda4",
            shape=(),
            initializer=tf.constant_initializer(0.0),
            trainable=True
        )

    def build(self, input_shape):
        self.encoder.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        f, cls = self.encoder(x,training=training)
        return f, cls
    
    def train_step(self, data):
        x, y = data
        y = tf.cast(y, tf.int32)
        with tf.GradientTape() as tape:
            f, cls = self(x, training=True)
            f = tf.nn.l2_normalize(tf.cast(f, tf.float32),axis=1)
            p = tf.nn.l2_normalize(tf.cast(self.prototypes, tf.float32),axis=1)
            Liccl = Loss_Component.intra_class_loss(f, y, p)
            Licmrl = Loss_Component.margin_loss(f,y,p,margin=0.15)
            Lpsl = Loss_Component.prototype_separation_loss(p,sigma=0.5)
            Luaer = Loss_Component.uncertainty_loss(f, p)
            Lce = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    tf.one_hot(y, depth=self.num_classes),
                    cls,
                    label_smoothing=0.0
                )
            )
            loss = tf.cond(
                tf.equal(self.stage, 0),
                lambda: 5.0 * Lce,
                lambda: self.adaptive_loss(
                    losses=[
                        Lce,
                        Liccl,
                        Licmrl,
                        Lpsl,
                        Luaer
                    ],
                    lambdas=[
                        tf.constant(3.0, dtype=tf.float32),
                        self.base_lambda1 +0.3 * tf.nn.softplus(self.lambda1),
                        self.base_lambda2 +0.3 * tf.nn.softplus(self.lambda2),
                        self.base_lambda3 +0.1 * tf.nn.softplus(self.lambda3),
                        self.base_lambda4 +0.05 * tf.nn.softplus(self.lambda4)
                    ]
                )
            )
        grads = tape.gradient(loss,self.trainable_variables)
        grads = [
            tf.clip_by_norm(g, 5.0)
            if g is not None else None
            for g in grads
        ]
        
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables)
        )
        p_old = tf.cast(
            self.prototypes,
            tf.float32
        )
        
        new_proto_list = []
        
        for c in range(self.num_classes):
        
            mask = tf.equal(y, c)
            class_feat = tf.boolean_mask(f,mask)
            class_feat = tf.cast(class_feat,tf.float32)
            class_mean = tf.cond(
                tf.shape(class_feat)[0] > 0,
                lambda: tf.reduce_mean(class_feat,axis=0),
                lambda: p_old[c]
            )
            updated = (
                0.7 * p_old[c] +
                0.3 * tf.stop_gradient(class_mean)
            )    
            new_proto_list.append(updated)
        new_prototypes = tf.stack(new_proto_list)
        new_prototypes = tf.nn.l2_normalize(new_prototypes,axis=1)
        self.prototypes.assign(new_prototypes)
        pred = tf.argmax(cls,axis=1,output_type=tf.int32)
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(pred, y),
                tf.float32
            )
        )
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.lce_tracker.update_state(Lce)
        self.liccl_tracker.update_state(Liccl)
        self.licmrl_tracker.update_state(Licmrl)
        self.lpsl_tracker.update_state(Lpsl)
        self.luaer_tracker.update_state(Luaer)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
            "Lce": Lce,
            "Liccl": Liccl,
            "Licmrl": Licmrl,
            "Lpsl": Lpsl,
            "Luaer": Luaer
        }

    def test_step(self, data):
        x, y = data
        y = tf.cast(y, tf.int32)
        f, cls = self(x, training=False)
        f = tf.nn.l2_normalize(tf.cast(f, tf.float32),axis=1)
        p = tf.nn.l2_normalize(
            tf.cast(self.prototypes, tf.float32),
            axis=1
        )
        Liccl = Loss_Component.intra_class_loss(f, y, p)
        Licmrl = Loss_Component.margin_loss(f,y,p,margin=0.15)
        Lpsl = Loss_Component.prototype_separation_loss(p,sigma=0.5)
        Luaer = Loss_Component.uncertainty_loss(f, p)
        Lce = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.one_hot(y, depth=self.num_classes),
                cls,
                label_smoothing=0.0
            )
        )
        loss = tf.cond(
            tf.equal(self.stage, 0),
            lambda: 5.0 * Lce,
            lambda: self.adaptive_loss(
                losses=[
                    Lce,
                    Liccl,
                    Licmrl,
                    Lpsl,
                    Luaer
                ],
                lambdas=[
                    tf.constant(3.0, dtype=tf.float32),
                    self.base_lambda1 + 0.3 * tf.nn.softplus(self.lambda1),
                    self.base_lambda2 + 0.3 * tf.nn.softplus(self.lambda2),
                    self.base_lambda3 + 0.1 * tf.nn.softplus(self.lambda3),
                    self.base_lambda4 + 0.05 * tf.nn.softplus(self.lambda4)
                ]
            )
        )        
        pred = tf.argmax(cls,axis=1,output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.lce_tracker.update_state(Lce)
        self.liccl_tracker.update_state(Liccl)
        self.licmrl_tracker.update_state(Licmrl)
        self.lpsl_tracker.update_state(Lpsl)
        self.luaer_tracker.update_state(Luaer)

        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_tracker.result(),
            "Lce": Lce,
            "Liccl": Liccl,
            "Licmrl": Licmrl,
            "Lpsl": Lpsl,
            "Luaer": Luaer
        }
    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.acc_tracker,
            self.lce_tracker,
            self.liccl_tracker,
            self.licmrl_tracker,
            self.lpsl_tracker,
            self.luaer_tracker
        ]