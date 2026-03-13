from dataset.tiny_imagenet_data_loader import TinyImageNetDataLoader
from models.resnet import ResNetBuilder
from config import training_config
import tensorflow as tf
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true", help="Resume training from last best model")
args = parser.parse_args()

learning_rate = training_config.LEARNING_RATE

data_loader = TinyImageNetDataLoader(batch_size=training_config.BATCH_SIZE)
train_ds, val_ds = data_loader.get_train_val_dataset()

model = ResNetBuilder().build_resnet18()
if args.resume:
    print("Loading weights from previous best model")
    model.load_weights(training_config.CHECKPOINT_PATH, skip_mismatch=True)
    learning_rate = training_config.LEARNING_RATE * 0.3

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

print(model.summary())

early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=training_config.EARLY_STOP_PATIENCE,
    min_delta=0.001,
    restore_best_weights=True,
)

model_checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath=training_config.CHECKPOINT_PATH,
    monitor="val_loss",
    save_best_only=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=training_config.REDUCE_LR_PATIENCE,
    factor=0.3
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=training_config.EPOCHS,
    callbacks=[early_stop_callback, model_checkpointer],
)

ts = int(time.time())
model.save(training_config.MODEL_PATH.replace("timestamp", str(ts)), overwrite=True)
