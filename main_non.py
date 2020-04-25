import tensorflow as tf


image_data = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

train_data = image_data.flow_from_directory('dataset/train/', target_size=(150, 150), batch_size=32, color_mode='rgb',class_mode='binary')
val_data = image_data.flow_from_directory('dataset/test/', target_size=(150, 150), batch_size=32, color_mode='rgb',class_mode='binary')

IMG_SHAPE = (150, 150, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


base_model.trainable = False


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(1)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./covid_research_non', histogram_freq=1)

model.fit_generator(train_data, epochs=100, callbacks=[tensorboard], validation_data=val_data)

model.save("covid_research_non.h5")