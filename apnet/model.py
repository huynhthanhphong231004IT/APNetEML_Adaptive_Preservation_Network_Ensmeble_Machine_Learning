import os
import math
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Input
from tensorflow.keras.utils import register_keras_serializable

from .layers import ANASPReLU, AdaptiveLoss, PrototypeModel
from .losscomponent import Loss_Component
from .backbones import build_default_backbone
from .callbacks import (
    APNetStageCallback,
    SaveAPNetHistory,
    PlotAPNetHistory
)

@register_keras_serializable(package="apnet")
class APNet(tf.keras.Model):
    def __init__(
        self,
        num_classes: int,
        input_shape: tuple = (64, 64, 3),
        embedding_dim: int = 512,
        backbone: tf.keras.Model = None,
        warmup_epochs: int = 25,
        Frozen: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.input_shape_val = input_shape
        self.warmup_epochs = warmup_epochs
        self.Frozen = Frozen

        self.stage = self.add_weight(
            name="stage", shape=(), dtype=tf.int32,
            initializer=tf.constant_initializer(0),
            trainable=False
        )

        backbone_feat = (
            build_default_backbone(input_shape, Frozen=Frozen)
            if backbone is None
            else backbone
        )
        
        inp = Input(shape=input_shape)
        x = backbone_feat(inp)
        x = Dense(embedding_dim, use_bias=False)(x)
        x = BatchNormalization()(x)
        feature = ANASPReLU()(x)
        classifier_output = Dense(num_classes, activation="softmax")(feature)

        self.encoder = Model(inp, [feature, classifier_output], name="APNet_Encoder")

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

        self.base_lambda1, self.base_lambda2, self.base_lambda3, self.base_lambda4 = 1.0, 0.5, 0.3, 0.05
        self.lambda1 = self.add_weight(name="lambda1", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.lambda2 = self.add_weight(name="lambda2", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.lambda3 = self.add_weight(name="lambda3", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.lambda4 = self.add_weight(name="lambda4", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)

    def build(self, input_shape):
        super().build(input_shape)
        if hasattr(self.adaptive_loss, 'build'):
            self.adaptive_loss.build(None)

    def call(self, x, training=False):
        return self.encoder(x, training=training)

    def _compute_losses(self, f, cls, y):
        f = tf.nn.l2_normalize(tf.cast(f, tf.float32), axis=1)
        p = tf.nn.l2_normalize(tf.cast(self.prototypes, tf.float32), axis=1)

        Liccl = Loss_Component.intra_class_loss(f, y, p)
        Licmrl = Loss_Component.margin_loss(f, y, p, margin=0.15)
        Lpsl = Loss_Component.prototype_separation_loss(p, sigma=0.5)
        Luaer = Loss_Component.uncertainty_loss(f,p)
        Lce = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.one_hot(y, depth=self.num_classes), cls
            )
        )

        adaptive_val = self.adaptive_loss(
            losses=[Lce, Liccl, Licmrl, Lpsl, Luaer],
            lambdas=[
                tf.constant(3.0, dtype=tf.float32),
                self.base_lambda1 + 0.3 * tf.nn.softplus(self.lambda1),
                self.base_lambda2 + 0.3 * tf.nn.softplus(self.lambda2),
                self.base_lambda3 + 0.1 * tf.nn.softplus(self.lambda3),
                self.base_lambda4 +0.05 * tf.nn.softplus(self.lambda4)
            ]
        )
        loss = tf.cond(
            tf.equal(self.stage, 0),
            lambda: 5.0 * Lce + 0.0 * adaptive_val,
            lambda: adaptive_val
        )
        return loss, Lce, Liccl, Licmrl, Lpsl, Luaer

    def train_step(self, data):
        x, y = data
        y = tf.cast(y, tf.int32)

        with tf.GradientTape() as tape:
            f, cls = self(x, training=True)
            loss, Lce, Liccl, Licmrl, Lpsl, Luaer = self._compute_losses(f, cls, y)

        grads = tape.gradient(loss, self.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        p_old = tf.cast(self.prototypes, tf.float32)
        f_norm = tf.nn.l2_normalize(tf.cast(f, tf.float32), axis=1)
        
        new_proto_list = []
        for c in range(self.num_classes):
            mask = tf.equal(y, c)
            class_feat = tf.boolean_mask(f_norm, mask)
            class_mean = tf.cond(
                tf.shape(class_feat)[0] > 0,
                lambda: tf.reduce_mean(class_feat, axis=0),
                lambda: p_old[c]
            )
            updated = 0.7 * p_old[c] + 0.3 * tf.stop_gradient(class_mean)
            new_proto_list.append(updated)

        new_prototypes = tf.nn.l2_normalize(tf.stack(new_proto_list), axis=1)
        self.prototypes.assign(new_prototypes)

        pred = tf.argmax(cls, axis=1, output_type=tf.int32)
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
            "Luaer": Luaer,

        }

    def test_step(self, data):
        x, y = data
        y = tf.cast(y, tf.int32)
        f, cls = self(x, training=False)
        
        loss, Lce, Liccl, Licmrl, Lpsl, Luaer = self._compute_losses(f, cls, y)
        pred = tf.argmax(cls, axis=1, output_type=tf.int32)
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
            "Luaer": Luaer,
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def fit_dataset(
        self,
        train_data,
        val_data=None,
        epochs: int = 300,
        batch_size: int = 64,
        learning_rate = 1e-5,
        weight_decay: float = 1e-4,
        save_dir: str = "./output",
        callbacks: list = None,
        **kwargs
    ):
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(train_data, tuple):
            X_train, y_train = train_data
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
            total_steps = math.ceil(len(X_train) / batch_size) * epochs
        elif isinstance(train_data, tf.data.Dataset):
            train_ds = train_data
            try:
                total_steps = len(train_ds) * epochs
            except TypeError:
                total_steps = kwargs.pop("total_steps", 10000)
        else:
            raise ValueError("train_data phải là Tuple (X, y) hoặc tf.data.Dataset")

        if isinstance(val_data, tuple):
            X_val, y_val = val_data
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            val_ds = val_data

        if isinstance(learning_rate, (float, int)):
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=float(learning_rate),
                decay_steps=total_steps,
                alpha=1e-2
            )
        else:
            lr_schedule = learning_rate

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            clipnorm=0.5
        )
        self.compile(optimizer=optimizer, run_eagerly=True)

        checkpoint_path = os.path.join(save_dir, "APNet_best.weights.h5")
        history_path = os.path.join(save_dir, "training_history.csv")

        default_callbacks = [
            APNetStageCallback(switch_epoch=self.warmup_epochs),
            SaveAPNetHistory(filepath=history_path),
            PlotAPNetHistory(save_dir=save_dir),

            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                mode="max",
                verbose=1
            )
        ]

        if callbacks:
            default_callbacks.extend(callbacks)

        history = self.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=default_callbacks,
            **kwargs
        )

        self.save_weights(os.path.join(save_dir, "APNet_final.weights.h5"))
        np.save(os.path.join(save_dir, "prototypes.npy"), self.prototypes.numpy())
        
        adaptive_weights = {}
        if hasattr(self.adaptive_loss, 'log_vars'):
            log_vars = self.adaptive_loss.log_vars
            if isinstance(log_vars, (list, tuple)):
                adaptive_weights = {f"log_var_{i}": float(w.numpy()) for i, w in enumerate(log_vars)}
            else:
                adaptive_weights = {f"log_var_{i}": float(val) for i, val in enumerate(log_vars.numpy())}
                
        joblib.dump(adaptive_weights, os.path.join(save_dir, "adaptive_loss_weights.pkl"))

        print(f"\n >>> [APNet] Training Complete! Artifacts saved to: {save_dir}")
        return history