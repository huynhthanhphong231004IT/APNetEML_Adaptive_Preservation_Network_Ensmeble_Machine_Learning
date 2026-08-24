import tensorflow as tf
import pandas as pd
from .layers import ANASPReLU

class APNetStageCallback(tf.keras.callbacks.Callback):
    def __init__(self, switch_epoch=25):
        super().__init__()
        self.switch_epoch = switch_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.switch_epoch:
            self.model.stage.assign(tf.constant(1, dtype=tf.int32))
            print(f"\n [APNet] Switched to Stage 1 (Full Adaptive Prototype Loss) at Epoch {epoch + 1}")

class SaveAPNetHistory(tf.keras.callbacks.Callback):
    def __init__(self, filepath="/kaggle/working/training_history.csv"):
        super().__init__()
        self.filepath = filepath
        self.history_data = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0)),
            "accuracy": float(logs.get("accuracy", 0)),
            "Liccl": float(logs.get("Liccl", 0)),
            "Licmrl": float(logs.get("Licmrl", 0)),
            "Lpsl": float(logs.get("Lpsl", 0)),
            "val_loss": float(logs.get("val_loss", 0)),
            "val_accuracy": float(logs.get("val_accuracy", 0)),
            "lambda1": float(self.model.lambda1.numpy()),
            "lambda2": float(self.model.lambda2.numpy()),
            "lambda3": float(self.model.lambda3.numpy()),
        }
        
        if hasattr(self.model, 'adaptive_loss') and hasattr(self.model.adaptive_loss, 'log_vars'):
            log_vars = self.model.adaptive_loss.log_vars
            if isinstance(log_vars, (list, tuple)):
                for i, w in enumerate(log_vars):
                    row[f"log_var_{i}"] = float(w.numpy())
            else:
                log_vars_vals = log_vars.numpy()
                for i, val in enumerate(log_vars_vals):
                    row[f"log_var_{i}"] = float(val)
            
        for layer in self.model.encoder.layers:
            if isinstance(layer, ANASPReLU):
                row["anasp_lambda"] = float(layer.lmbda.numpy())
                row["anasp_alpha"] = float(layer.alpha.numpy())
                row["anasp_beta"] = float(layer.beta.numpy())
                row["anasp_gamma"] = float(layer.gamma.numpy())
                row["anasp_tau"] = float(layer.tau.numpy())
                
        self.history_data.append(row)
        pd.DataFrame(self.history_data).to_csv(self.filepath, index=False)