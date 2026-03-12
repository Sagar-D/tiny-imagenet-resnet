from dataset.tiny_imagenet_data_loader import TinyImageNetDataLoader
from models.resnet import ResNetBuilder
from config import training_config
import tensorflow as tf
import time

data_loader = TinyImageNetDataLoader(batch_size=training_config.BATCH_SIZE)
train_ds, val_ds = data_loader.get_train_val_dataset()

model = ResNetBuilder().build_resnet18()

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

print(model.summary())

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=training_config.PATIENCE,
    min_delta=0.001,
    restore_best_weights=True,
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=training_config.EPOCHS,
    callbacks=[early_stop_callback],
)

ts = int(time.time())
model.save(training_config.MODEL_PATH.replace("timestamp", str(ts)), overwrite=True)
