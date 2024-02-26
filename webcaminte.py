from prediction import generate_caption
import cv2
from PIL import Image
import pyttsx3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import json

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
        tokenizer = Tokenizer()
        print(tokenizer_data)
        tokenizer.word_index = tokenizer_data['word_index']
        tokenizer.index_word = tokenizer_data['index_word']
    return tokenizer

text_speech = pyttsx3.init()

cap = cv2.VideoCapture("../videos/pexels-tony-schnagl-6338352 (Original).mp4")
# cap = cv2.VideoCapture(0)
cap.set(3,1200)
cap.set(4,720)

# Load the trained model and tokenizer
model_path = './best_model (1).h5'
tokenizer_path = './tokenizer_path.json'
model = load_model(model_path)
tokenizer = load_tokenizer(tokenizer_path)
max_length = 35

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1200,720))
    
    # Press 'q' to exit the loop and close the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Press 'c' to capture the frame and generate a caption
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        # Convert OpenCV frame to PIL Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(image)
        
        # Generate and print the caption
        caption = generate_caption(model, tokenizer, max_length, image)
        caption = caption[8:-6]
        
        text_speech.say(caption)
        text_speech.runAndWait()
        print("Generated Caption:", caption)
        cv2.putText(frame, caption, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Webcam', frame)

        

cap.release()
cv2.destroyAllWindows()