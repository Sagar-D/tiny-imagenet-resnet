import kagglehub
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf


class TinyImageNetDataLoader:

    def __init__(self, batch_size=128):
        
        self.batch_size = batch_size

        self.dataset_path = None
        self.labels = []
        self.label_to_index_map = {}
        self.train_dataset = None
        self.val_dataset = None
        self._download_dataset()
        self._build_train_dataset()

    def _download_dataset(self) -> None:

        dataset_path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
        dataset_path = Path(os.path.join(dataset_path, "tiny-imagenet-200"))
        print("Path to dataset files:", dataset_path)
        self.dataset_path = dataset_path

    def _load_image(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (64, 64))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def _build_train_dataset(self) -> None:

        training_images = []
        training_labels = []
        for image_path in (self.dataset_path / "train").glob("*/*/*.JPEG"):
            training_images.append(str(image_path))
            training_labels.append(image_path.parent.parent.name)

        self.labels = sorted(set(training_labels))
        self.label_to_index_map = {}
        self.index_to_label_map = {}
        for i, label in enumerate(self.labels) :
            self.label_to_index_map[label] = i
            self.index_to_label_map[i] = label

        training_labels = [self.label_to_index_map[label] for label in training_labels]

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (training_images, training_labels)
        )
        train_dataset = train_dataset.map(
            self._load_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(10000)
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        self.train_dataset = train_dataset

    def _build_val_dataset(self) -> None:

        val_annotations = pd.read_csv(
            (self.dataset_path / "val/val_annotations.txt"),
            sep="\t",
            header=None,
            names=["image_name", "label", "bx1", "by1", "bx2", "by2"],
        )

        val_images = []
        val_labels = []
        for row in val_annotations[["image_name","label"]].itertuples() :
            val_images.append(str(self.dataset_path / "val" / "images" / row.image_name))
            val_labels.append(self.label_to_index_map[row.label])

        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.map(
            self._load_image, num_parallel_calls=tf.data.AUTOTUNE
        )
        val_dataset = val_dataset.cache()
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        self.val_dataset = val_dataset

    def get_train_dataset(self) -> tf.data.Dataset:
        if not self.train_dataset:
            self._build_train_dataset()
        return self.train_dataset

    def get_val_dataset(self) -> tf.data.Dataset:
        if not self.val_dataset:
            self._build_val_dataset()
        return self.val_dataset

    def get_train_val_dataset(
        self,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        return (self.get_train_dataset(), self.get_val_dataset())


if __name__ == "__main__" :

    loader = TinyImageNetDataLoader()
    train_ds, val_ds = loader.get_train_val_dataset()
    for images, labels in val_ds.take(1):
        for image, label in zip(images[:10],labels[:10]) :
            print(f"{loader.index_to_label_map[int(label)]}")