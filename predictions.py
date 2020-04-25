import tensorflow as tf # for deep learning

model =  tf.keras.models.load_model("covid-19_dataset/covid_research_non.h5")

print(model.weights)