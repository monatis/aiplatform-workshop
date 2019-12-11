import tensorflow as tf
import numpy as np
import os

 # Utility functions to apply data augmentations.
 # some of the functions directly borrowed from https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/

def flip(x):
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def color(x):
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x

def rotate(x):
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def zoom(x):
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.75, lambda: x, lambda: random_crop(x))

def load_image(image_path, args, augment=False):
    label = encode_label(image_path, args["class_names"])
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if augment:
        augmentations = [flip, color, zoom, rotate]
        for f in augmentations:
            img = tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(img), lambda: img)
        img = tf.clip_by_value(img, 0, 1)
    img = tf.image.resize(img, (args["input_size"], args["input_size"]))
    img = tf.multiply(img, 1./255.)
    return img, label

def encode_label(filepath, classes):
    label = tf.strings.split(filepath, os.path.sep, result_type='RaggedTensor')[-2]
    label = label == classes
    return tf.cast(label, tf.int32)

def create_dataset(args):
    train_ds = tf.data.Dataset.list_files(os.path.join(args["train_dir"], '*', '*.jpg')).map(lambda x: load_image(x, args, augment=True), num_parallel_calls=8)
    val_ds = tf.data.Dataset.list_files(os.path.join(args["validation_dir"], '*', '*.jpg')).map(lambda x: load_image(x, args, augment=False), num_parallel_calls=8)
    return train_ds, val_ds
        