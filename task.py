from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

weight_decay = 1e-4


def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x


def vgg8(args):
    input = Input(shape=(args.input_size, args.input_size, 3))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = Dense(args.num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)

    return Model(input, output)


train_aug = ImageDataGenerator(rescale=1./255,  # rescale the tensor values to [0,1]
                               shear_range=0.2,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               # rotation_range=90,
                               zoom_range=0.2,
                               horizontal_flip=True)

validation_aug = ImageDataGenerator(rescale=1./255.)


def generate_train_generator(args):
    train_gen = train_aug.flow_from_directory(
        args.train_dir,
        target_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True)
    while True:
        batch = train_gen.next()
        yield batch


def generate_validation_generator(args):
    validation_gen = validation_aug.flow_from_directory(
        args.validation_dir,
        target_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False)
    while True:
        batch = validation_gen.next()
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_size", default=150, type=int)
    ap.add_argument("--num_features", default=64, type=int)
    ap.add_argument("--num_classes", default=2, type=int)
    ap.add_argument("--train_dir", required=True, type=str)
    ap.add_argument("--validation_dir", required=True, type=str)
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--num_train_images", default=2000, type=int)
    ap.add_argument("--num_validation_images", default=1000, type=int)
    ap.add_argument("--job-dir", default="./", type=str)
    args = ap.parse_args()
    model = vgg8(args)

    model.compile(loss="categorical_crossentropy", optimizer='adam')
    train_images = generate_train_generator(args)
    validation_images = generate_validation_generator(args)

    model.fit_generator(train_images,
                        steps_per_epoch=args.num_train_images//args.batch_size,
                        validation_data=validation_images,
                        validation_steps=args.num_validation_images//args.batch_size,
                        epochs=args.epochs,
                        verbose=1)
                        
    model.save(args.job_dir)

if __name__ == "__main__":
    main()