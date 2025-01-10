import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import glob
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import imgaug.augmenters as iaa
from imgaug import parameters as iap
import os
def PNN():
    for dirname, _, filenames in os.walk('input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    data_dir = "The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset"
    label_dirs = ['Bengin cases', 'Malignant cases', 'Normal cases']
    out_dir = 'working'
    train_ratio = 0.5
    val_ratio = 0.25
    test_ratio = 0.25
    label_dirs
    import shutil
    for label in label_dirs:
        img_paths = glob.glob(os.path.join(data_dir, label, "*.jpg"))
        random_seed = 42  
        train_paths, rest_paths = train_test_split(img_paths, train_size=train_ratio, random_state=random_seed)
        val_paths, test_paths = train_test_split(rest_paths, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed)
        for phase, paths in zip(["train", "val", "test"], [train_paths, val_paths, test_paths]):
            phase_dir = os.path.join(out_dir, phase)
            if not os.path.exists(phase_dir):
                os.makedirs(phase_dir)
            label_dir = os.path.join(phase_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            for path in paths:
                dst_path = os.path.join(label_dir, os.path.basename(path))
                shutil.copy(path, dst_path)  

    train_benign = r'The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\Benign cases'
    train_malignant = r'The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\Malignant cases'
    train_normal = r'The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset\Normal cases'
    val_benign = 'working/val/Benign cases'
    val_malignant = 'working/val/Malignant cases'
    val_normal = 'working/val/Normal cases'
    def create_augmentation_seq():
        augmentation_seq = iaa.Sequential([
          iaa.Rot90([1, 2, 3]),
          iaa.Fliplr(0.5),
          iaa.Affine(translate_px=iap.DiscreteUniform(-10, 10), scale=(0.9, 1.1)),
          iaa.CropAndPad(px=(-10, 10)),
          iaa.AdditiveGaussianNoise(scale=(0, 30)),
          iaa.LinearContrast((0.9, 1.1)),
        ])
        return augmentation_seq
    def augment_and_save_images(input_path, output_path, num_augmented):
        img = Image.open(input_path)
        img_np = np.array(img)
        augmentation_seq = create_augmentation_seq()
        for i in range(num_augmented):
            aug_img_np = augmentation_seq(image=img_np)
            aug_img = Image.fromarray(aug_img_np)
            save_path = os.path.join(output_path, f"augmented_{i}_{os.path.basename(input_path)}")
            aug_img.save(save_path)
    class_folders = [train_benign, train_malignant, train_normal]
    for class_folder in class_folders:
        image_paths = glob.glob(os.path.join(class_folder, "*.jpg"))
        for img_path in image_paths:
            augment_and_save_images(img_path, class_folder, num_augmented=4)
    class_folders = [val_benign, val_malignant, val_normal]
    for class_folder in class_folders:
        image_paths = glob.glob(os.path.join(class_folder, "*.jpg"))
        for img_path in image_paths:
            augment_and_save_images(img_path, class_folder, num_augmented=4)
    train_dir = 'working/train'
    test_dir = 'working/test'
    val_dir = 'working/val'
    image_size = 512
    batch_size = 32
    train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                     image_size=(image_size, image_size),
                                                                     batch_size=batch_size,
                                                                     shuffle=True)
    val_data = tf.keras.preprocessing.image_dataset_from_directory(val_dir,
                                                                   image_size=(image_size, image_size),
                                                                   batch_size=batch_size,
                                                                   shuffle=False)
    test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                    image_size=(image_size, image_size),
                                                                    batch_size=batch_size,
                                                                    shuffle=False)
    for image_batch, labels_batch in train_data:
      print(image_batch.shape)
      print(labels_batch.shape)
      break
    class_names = train_data.class_names
    print(class_names)
    plt.figure(figsize = (10,10))
    for images, labels in train_data.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
    image_batch
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))
    def standard(image, label):
        image = tf.cast(image, tf.float32)
        image = (image/127.5) -1   
        return image, label
    train = train_data.map(standard)
    validation = val_data.map(standard)
    for normal_batch, train_label in train:
      print(normal_batch.shape)
      print(train_label.shape)
      break
    train_label
    for val_batch, val_label in validation:
      print(normal_batch.shape)
      print(train_label.shape)
      break
    normal_image = normal_batch[0]
    print(np.min(normal_image), np.max(normal_image))
    val_image = val_batch[0]
    print(np.min(val_image), np.max(val_image))
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
    IMG_SHAPE = (image_size, image_size, 3)
    model = Sequential([
        Conv2D(32, (3,3), activation = 'relu', input_shape = IMG_SHAPE),
        MaxPooling2D(pool_size = (2, 2)),
        Conv2D(64, (3,3), activation = 'relu'),
        MaxPooling2D(pool_size = (2 ,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'), 
        Dense(3, activation='softmax') 
    ])
    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train,
                        batch_size=32,
                        epochs=10,
                        validation_data=validation)
    def plot_graghs(history, metric):
      plt.plot(history.history[metric])
      plt.plot(history.history['val_'+metric], '')
      plt.xlabel('Epochs')
      plt.ylabel(metric)
      plt.legend([metric, 'val_'+metric])
      plt.show()
    plot_graghs(history, 'accuracy')
    plot_graghs(history, 'loss')
    test = test_data.map(standard)
    result = model.evaluate(test)
    print("Test loss, Test accuracy : ", result)
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_dir = 'working/train'
    validation_dir = 'working/val'
    test_dir = 'working/test'
    image_height = 150
    image_width = 150
    batch_size = 32
    num_epochs = 10
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(image_height, image_width),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')  

    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            target_size=(image_height, image_width),
                                                            batch_size=batch_size,
                                                            class_mode='categorical')  

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=(image_height, image_width),
                                                      batch_size=batch_size,
                                                      class_mode='categorical') 

    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=num_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples // batch_size)

    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
    print(f'Test accuracy: {test_acc}')
    predictions = model.predict(test_generator, steps=test_generator.samples // batch_size)
    class_labels = train_generator.class_indices
    for i, prediction in enumerate(predictions):
        predicted_class = list(class_labels.keys())[list(class_labels.values()).index(tf.argmax(prediction))]
        print(f"Sample {i + 1}: Predicted Class - {predicted_class}")

