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
<p align="center">
   <b>Presional link Information</b>
</p>

<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Facbook: https://www.facebook.com/share/1Md5MZhbkJ/ <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Kaggle: https://www.kaggle.com/reorioll <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Youtobe: https://www.youtube.com/@ReoRioll-2304CICTCTU <br>
</p>
<br>

<h3 align="left">
  <span style="color:#8B4513;">
    <b>Guide to training APNetEML with CIFAR-10 on Google Colaboratory</b>
  </span>
</h3>

<p>
  Step 1. Clone module from git
</p>

```python
!git clone https://github.com/huynhthanhphong231004IT/Adaptive_Preservation_Network.git
```

<p>
  Step 2. Change the current working module directory in Google Colab.
</p>

```python
%cd /content/Adaptive_Preservation_Network
```

<p>
  Step 3. Install all Python libraries listed in the requirements.txt file.
</p>

```python
!pip install -r requirements.txt
```

<p>
  Step 4. Download the CIFAR-10 training dataset.
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
  Step 5. Cosine Annealing Normalization for Learning Rate
</p>
<img src="https://latex.codecogs.com/svg.image?%5Ceta_t%20%3D%20%5Ceta_%7Bmin%7D%20%2B%20%5Cfrac%7B1%7D%7B2%7D%28%5Ceta_%7Bmax%7D%20-%20%5Ceta_%7Bmin%7D%29%5Cleft%281%20%2B%20%5Ccos%5Cleft%28%5Cfrac%7Bt%7D%7BT%7D%5Cpi%5Cright%29%5Cright%29%2C%20%5Ctext%7Bpatience%7D%20%3D%203" width="450" style="vertical-align: middle;">


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
  Step 6. Configure training parameters
</p>

```python
import tensorflow as tf
from apnet import APNet
ETA_MAX = 1e-5
ETA_MIN = 1e-7
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
  Step 7. Deep feature training on APNetCNN
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
> **`num_classes`**: The total number of classes (labels) in the dataset.
>
> **`input_shape`**: The input image dimensions in RGB format.
>
> **`embedding_dim`**: The dimensionality of the feature vector (embedding) passed to the PD-Loss (recommended: `512`, minimum: `256`).
>
> **`backbone`**: No external pretrained backbone is used (`None`).
>
> **`warmup_epochs`**: The number of initial epochs used for the phase transition in the PD-Loss function.
>
> **`Frozen`**: Set to `True` to freeze part of the backbone (Conv2D layers with 216 and 512 filters). Set to `False` to train all backbone layers.

<h3 align="left">
  <span style="color:#8B4513;">
    <b>Theoretical framework of the proposed study</b>
  </span>
</h3>

<div align="left">
<br>
<p>
  <a href="docs/APNetEML_Theory.md">
    <font size="5"><b>I. Proposed Theory</b></font>
  </a>
  <br>
  <font size="3">Theoretical foundation of APNet-EML architecture: (1) Activation functions including SNReLU and ANASPReLU;(2) Mix loss function with 5 baseline losses of Deep Lerning is PD-Loss; (3) Ensemble Machine Learing with 4 algorithms including SVM, Random Forest, Logistic Regestion, MLP in Vote Soft</font>
</p>
<br>
<p>
  <a href="docs/APNetEML_Results.md">
    <font size="5"><b>II. Experimental Results</b></font>
  </a>
  <br>
  <font size="3">Experimental evaluation with performance analysis and model generalization including effecting activations, effecting backprop different components of PD-Loss, comparing baseline models. Experimental datasets with Military Region 9 Museum of Can Tho, CIFAR-10, CIFAR-100, ... </font>
</p>

</div>
