import numpy as np
from apnet import APNet

X_train = np.random.rand(1000, 64, 64, 3).astype(np.float32)
y_train = np.random.randint(0, 10, size=(1000,))
X_val = np.random.rand(200, 64, 64, 3).astype(np.float32)
y_val = np.random.randint(0, 10, size=(200,))

model = APNet(
    num_classes=10,
    input_shape=(64, 64, 3),
    embedding_dim=1024,
    backbone=None,
    warmup_epochs=5,
    Frozen=False
)

model.fit_dataset(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    learning_rate=1e-5,
    save_dir="./my_numpy_experiment"
)