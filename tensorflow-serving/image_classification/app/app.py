import sys
import os
import json
import string
import time
import io
import requests
import numpy as np

# Importing TensorFlow
import tensorflow as tf
import tensorflow_hub as hub

# Loading model
# model_path = './model/'
# loaded_model = tf.saved_model.load(model_path)
classifier_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
  hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,))
])

def make_dataset(batch_size, size):
    image_shape = (size, size, 3)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

def handler(event, context):
    batch_size = event['batch_size']
    size = event['size']
    data,image_shape = make_dataset(batch_size, size)

    # Executing inference.
    start_time = time.time()
    result = classifier.predict(data)
    end_time = time.time()

    obj = {
        "result":result
    }    

    return {
        'statusCode': 200,
        'body': json.dumps(obj)
    }
