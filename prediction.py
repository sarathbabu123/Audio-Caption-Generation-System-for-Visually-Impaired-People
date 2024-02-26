from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import json
import numpy as np
import cv2


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
        tokenizer = Tokenizer()
        print(tokenizer_data)
        tokenizer.word_index = tokenizer_data['word_index']
        tokenizer.index_word = tokenizer_data['index_word']
    return tokenizer

# def extract_features(filename):
#     model = VGG16()
#     model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
#     image = load_img(filename, target_size=(224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#     feature = model.predict(image, verbose=0)
#     return feature

def extract_features(filename):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # image = load_img(filename, target_size=(224, 224))
    # image = img_to_array(image)
    image = cv2.resize(filename, (224,224))
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature


def generate_caption(model, tokenizer, max_length, image_filename):
    photo = extract_features(image_filename)
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# # Load the trained model and tokenizer
# model_path = './best_model.h5'
# tokenizer_path = './tokenizer_path.json'
# model = load_model(model_path)
# tokenizer = load_tokenizer(tokenizer_path)
# max_length = 35
# # Example usage
# image_filename = './10815824_2997e03d76.jpg'
# caption = generate_caption(model, tokenizer, max_length, image_filename)
# print('Generated Caption:', caption)
