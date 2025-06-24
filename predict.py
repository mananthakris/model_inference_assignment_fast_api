#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DogCatClassifier:
    def __init__(self, filename):
        self.filename = filename

    def prediction_dog_cat(self):
        # Load model
        model = load_model('model.h5')

        # Load and preprocess the image
        imagename = self.filename
        test_image = load_img(imagename, target_size=(64, 64))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Rescale pixel values to the range [0, 1] as done during training

        # Predict the class (0 for cat, 1 for dog)
        result = model.predict(test_image)

        if result[0][0] > 0.5:
            prediction = 'dog'
        else:
            prediction = 'cat'

        return [{"image": prediction}]

# Example usage:
# classifier = DogCatClassifier('path/to/your/image.jpg')
# prediction_result = classifier.prediction_dog_cat()
# print(prediction_result)
