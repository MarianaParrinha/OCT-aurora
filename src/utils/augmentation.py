# src/utils/augmentation.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator


#gerador de data augmentation para imagens e máscaras
def create_augmentation_generators(X_train, y_train, batch_size=8, seed=42):
    """
    Cria geradores sincronizados para imagens e máscaras.
    O mesmo seed garante que a transformação aplicada à imagem
    é a mesma aplicada à máscara correspondente.
    """

    data_gen_args = dict(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow(
        X_train,
        batch_size=batch_size,
        seed=seed
    )

    mask_generator = mask_datagen.flow(
        y_train,
        batch_size=batch_size,
        seed=seed
    )

    train_generator = zip(image_generator, mask_generator)

    return train_generator


#caso seja preciso separar a definição dos datagens
def get_image_and_mask_datagens():
    data_gen_args = dict(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    return image_datagen, mask_datagen


#função alternativa para criar os generators a partir dos datagens
def make_generators(X_train, y_train, batch_size=8, seed=42):
    image_datagen, mask_datagen = get_image_and_mask_datagens()

    image_generator = image_datagen.flow(
        X_train,
        batch_size=batch_size,
        seed=seed
    )

    mask_generator = mask_datagen.flow(
        y_train,
        batch_size=batch_size,
        seed=seed
    )

    train_generator = zip(image_generator, mask_generator)

    return train_generator