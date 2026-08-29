<h2 align="center">
  Author: Huynh Thanh Phong (ReoRioll)
</h2>

<p align="center">
   Computer Science of College of Information and Communication Technology of Can Tho University (Course 48)<br>
</p>

<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<b>Researchs:</b> Artificial Intelligence in Education - Mathematics in Deep Learning and Machine Learning<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<mark><b><b>Name Project:</b></b> </mark> "Hybrid Model APNetEML" Adaptive Preservation Network - Ensemble Machine Learning<br>


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<b>Timeline:</b> 03/2025 – 08/2026 at Computer science department
</p>
<br>

<h3 align="left">
  <span style="color:#8B4513;">
    <b>Hướng dẫn huấn luyện APNetEML với CIFAR-10 trên Google Colaboratory</b>
  </span>
</h3>

<p>
  Bước 1. Clone module từ git về
</p>

```python
!git clone https://github.com/huynhthanhphong231004IT/Adaptive_Preservation_Network.git
```

<p>
  Bước 2. Chuyển thư mục module làm việc hiện tại trong Google Colab
</p>

```python
%cd /content/Adaptive_Preservation_Network
```

<p>
  Bước 3. Cài tất cả các thư viện Python trong file requirements.txt
</p>

```python
!pip install -r requirements.txt
```

<p>
  Bước 4. Tải tập dữ liệu huấn luyện CIFAR-10 
</p>

```python
import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = tf.image.resize(X_train, (64, 64)).numpy().astype(np.float32)
X_test  = tf.image.resize(X_test, (64, 64)).numpy().astype(np.float32)

y_train = y_train.flatten()
y_test = y_test.flatten()

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)
```

<p>
  Bước 5. Chuấn hóa Cosine Annealing cho Learning Rate
</p>

```python
class CosineLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, eta_max, eta_min, total_steps):
        super().__init__()
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        t = tf.minimum(step,tf.cast(self.total_steps, tf.float32))
        lr = (self.eta_min+ 0.5 * (self.eta_max - self.eta_min)* (1.0+ tf.cos((t / self.total_steps) * tf.constant(3.141592653589793))))
        return lr

    def get_config(self):
        return {
            "eta_max": self.eta_max,
            "eta_min": self.eta_min,
            "total_steps": self.total_steps
        }
```

<p>
  Bước 6. Thiết lập các thông số huấn luyện
</p>

```python
TOTAL_EPOCHS = 200
BATCH_SIZE = 64
STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
TOTAL_STEPS = TOTAL_EPOCHS * STEPS_PER_EPOCH
```

```python
lr_schedule = CosineLRSchedule(
    eta_max=ETA_MAX,
    eta_min=ETA_MIN,
    total_steps=TOTAL_STEPS
)
```

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=30,
    mode="max",
    restore_best_weights=True,
    verbose=1
)
```

<p>
  Bước 7. Huấn luyện đặc trưng sâu trên APNet
</p>


```python
model = APNet(
    num_classes=10,
    input_shape=(64, 64, 3),
    embedding_dim=1024,
    backbone=None,
    warmup_epochs=5,
    Frozen=False
)
```

```python
history = model.fit_dataset(
    train_data=(X_train, y_train),
    val_data=(X_test, y_test),
    epochs = TOTAL_EPOCHS,
    batch_size = BATCH_SIZE,
    learning_rate = lr_schedule,
    save_dir="my_path",
    callbacks=[early_stopping]
)
```

> [!NOTE]
> **`num_classes`**: Tổng số nhãn (classes) của tập dữ liệu.
>
> **`input_shape`**: Kích thước ảnh đầu vào RGB.
>
> **`embedding_dim`**: Số chiều của vector đặc trưng (embedding) truyền vào PD-Loss (gợi ý: `512`, min `256`).
>
> **`backbone`**: Không sử dụng backbone có sẵn bên ngoài (`None`).
>
> **`warmup_epochs`**: Số epoch đầu dùng cho giai đoạn chuyển pha trong PD-Loss function.
>
> **`Frozen`**: `True` nếu muốn đóng băng (freeze) một phần backbone (tầng Conv 216 và 512). `False` nếu sử dụng toàn bộ tầng backbone.

<div align="left">
<p>
  <a href="docs/APNetEML_Theory.md">
    <font size="5"><b>I. Proposed Theory</b></font>
  </a>
  <br>
  <font size="3">Theoretical foundation of APNet-EML architecture: (1) Activation functions including SNReLU and ANASPReLU;(2) Mix loss function with 5 baseline losses of Deep Lerning is PD-Loss; (3) Ensemble Machine Learing with 4 algorithms including SVM, Random Forest, Logistic Regestion, MLP in Vote Soft</font>
</p>

<p>
  <a href="docs/APNetEML_Results.md">
    <font size="5"><b>II. Experimental Results</b></font>
  </a>
  <br>
  <font size="3">Experimental evaluation with performance analysis and model generalization including effecting activations, effecting backprop different components of PD-Loss, comparing baseline models. Experimental datasets with Military Region 9 Museum of Can Tho, CIFAR-10, CIFAR-100, ... </font>
</p>

</div>
