from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import regularizers
import tensorflow as tf
import argparse
from . import utils

weight_decay = 1e-4


def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x


def vgg8(args):
    input = Input(shape=(args["input_size"], args["input_size"], 3))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args["num_features"], kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = Dense(args["num_classes"], activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)

    return Model(input, output)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_size", default=150, type=int)
    ap.add_argument("--num_features", default=64, type=int)
    ap.add_argument("--num_classes", default=2, type=int)
    ap.add_argument("--train_dir", required=True, type=str)
    ap.add_argument("--validation_dir", required=True, type=str)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--job-dir", default="./", type=str)
    args = vars(ap.parse_args())
    args["class_names"] = ['dogs', 'cats']
    model = vgg8(args)

    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['acc'])
    train_dataset, validation_dataset = utils.create_dataset(args)

    model.fit(train_dataset.shuffle(1000).batch(args["batch_size"]),
                        validation_data=validation_dataset.batch(args["batch_size"]),
                        epochs=args["epochs"],
                        verbose=1)
                        
    model.save('model.h5')
    if args["job_dir"].startswith("gs://"):
        with open('model.h5', "rb") as local_file:
            with tf.gfile.open(args["job_dir"] + "/model.h5", "wb") as gcs_file:
                gcs_file.write(local_file.read())
        
if __name__ == "__main__":
    main()