# !/usr/bin/env python3

"""
Character Recognition

This code provides a character recognition system that can recognize characters in an image using a machine learning model. It includes functionalities for dataset collection, preprocessing, feature extraction, model training, character recognition, language modeling, and PDF output.

Author: CireWire
License: MIT License
Date: 05/26/2023

Requirements:
- Python 3.x
- OpenCV (cv2)
- NumPy
- scikit-learn
- python-pptx
- reportlab
- argparse
- pytesseract
- spellchecker
- PyPDF2

Usage:
- Follow the instructions in the README file to set up the project.
- Adjust the code based on your requirements.
- Run the code using the main function and provide the necessary arguments.

Note: Make sure to install the required dependencies before running the code.

"""


import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Example SVM model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from spellchecker import SpellChecker
from PyPDF2 import PdfWriter
from reportlab.pdfgen import canvas
import argparse
import logging
import pytesseract

def collect_dataset(directory):
    """
    Collects a dataset of labeled character images from the specified directory.

    Args:
        directory (str): The directory path containing the character images.

    Returns:
        dataset (list): A list of preprocessed character images.
        labels (list): A list of corresponding labels for the character images.
    """
    dataset = []
    labels = []

    try:
        # Iterate through the directory containing character images
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                file_path = os.path.join(directory, filename)
                image = cv2.imread(file_path)

                if image is not None:
                    # Preprocess the image if needed (e.g., resize, convert to grayscale, normalize pixel values)
                    preprocessed_image = preprocess_image(image)

                    # Add the preprocessed image to the dataset
                    dataset.append(preprocessed_image)

                    # Extract the label from the filename or assign a label based on the directory structure
                    label = extract_label(filename)  # Implement this function based on your labeling scheme
                    labels.append(label)

    except FileNotFoundError:
        logging.error("Directory not found.")
        return None, None
    except Exception as e:
        logging.error("Error occurred while collecting the dataset: %s", str(e))
        return None, None

    return dataset, labels

def preprocess_image(image):
    """
    Preprocesses an image for character recognition.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        preprocessed_image (numpy.ndarray): The preprocessed image.
    """
    try:
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to convert the image into binary format
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        return binary_image
    except Exception as e:
        logging.error("Error occurred while preprocessing the image: %s", str(e))
        return None

def extract_label(filename):
    """
    Extracts the label from the filename or assigns a label based on the directory structure.

    Args:
        filename (str): The filename of the character image.

    Returns:
        label (str): The label of the character image.
    """
    # Extract label from the filename or assign a label based on the directory structure
    return os.path.splitext(filename)[0]  # Example: Use the filename without the extension as the label

def extract_features(segmented_characters):
    """
    Extracts features from the segmented characters.

    Args:
        segmented_characters (list): A list of segmented characters.

    Returns:
        features (list): A list of extracted features for the characters.
    """
    # Extract features from the segmented characters
    features = []

    for character in segmented_characters:
        # Extract features from each character
        feature = character.shape[0]  # Example: Use the height of the character as the feature
        features.append(feature)

    return features

def select_model():
    """
    Selects an appropriate model for character recognition.

    Returns:
        model: The selected machine learning or deep learning model.
    """
    # Select and return a model for character recognition
    return SVC()  # Example: Use Support Vector Machines (SVM)

def train_model(dataset, labels, model):
    """
    Trains the selected model using the dataset and labels.

    Args:
        dataset (list): The dataset of preprocessed character images.
        labels (list): The corresponding labels for the character images.
        model: The machine learning or deep learning model.

    Returns:
        None
    """
    # Train the model using the dataset and labels
    model.fit(dataset, labels)

def recognize_characters(image, model):
    """
    Recognizes characters in an image using the trained model.

    Args:
        image (numpy.ndarray): The input image.
        model: The trained machine learning or deep learning model.

    Returns:
        recognized_characters (list): A list of recognized characters.
    """
    # Perform character recognition using the trained model
    # Preprocess the input image, extract segmented characters, and classify them using the model
    segmented_characters = segment_characters(image)
    recognized_characters = model.predict(segmented_characters)

    return recognized_characters

def group_characters(segmented_characters, threshold):
    """
    Groups segmented characters together to reconstruct text lines or words.

    Args:
        segmented_characters (list): A list of segmented characters.
        threshold (int): The vertical threshold to determine line breaks.

    Returns:
        grouped_characters (list): A list of grouped characters.
    """
    grouped_characters = []

    # Sort the segmented characters vertically to group them into lines
    segmented_characters.sort(key=lambda c: cv2.boundingRect(c)[1])

    line_characters = [segmented_characters[0]]
    prev_y = cv2.boundingRect(segmented_characters[0])[1]

    for character in segmented_characters[1:]:
        x, y, _, _ = cv2.boundingRect(character)

        if y - prev_y < threshold:
            line_characters.append(character)
        else:
            # Sort the characters in each line horizontally to group them into words
            line_characters.sort(key=lambda c: cv2.boundingRect(c)[0])
            grouped_characters.append(line_characters)
            line_characters = [character]

        prev_y = y

    # Add the last line of characters
    line_characters.sort(key=lambda c: cv2.boundingRect(c)[0])
    grouped_characters.append(line_characters)

    return grouped_characters

def apply_language_model(grouped_characters):
    """
    Applies language modeling techniques to improve recognition accuracy and handle contextual errors.

    Args:
        grouped_characters (list): A list of grouped characters.

    Returns:
        predicted_words (list): A list of predicted words based on the grouped characters and context.
    """
    predicted_words = []

    # Apply language modeling techniques based on your requirements
    # You can use pre-trained language models, statistical language models, or custom models

    # Example:
    # Iterate over the grouped characters and perform language modeling
    for line_characters in grouped_characters:
        line_text = ""

        # Concatenate characters in each line to form text
        for character in line_characters:
            # Convert the character image to text using OCR or other character recognition techniques
            character_text = recognize_character(character)

            # Append the recognized character to the line text
            line_text += character_text

        # Apply language modeling techniques to the line text
        predicted_line = apply_language_model_to_line(line_text)

        # Split the line text into words and append to the predicted words list
        predicted_words.extend(predicted_line.split())

    return predicted_words

def recognize_character(character):
    """
    Recognizes a single character using OCR or other character recognition techniques.

    Args:
        character (numpy.ndarray): The segmented character image.

    Returns:
        recognized_text (str): The recognized text of the character.
    """
    # Convert the character image to grayscale
    gray_character = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance character visibility if needed
    _, binary_character = cv2.threshold(gray_character, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply OCR using pytesseract
    recognized_text = pytesseract.image_to_string(binary_character, config='--psm 10 --oem 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    return recognized_text


def apply_language_model_to_line(line_text):
    """
    Applies language modeling techniques to improve recognition accuracy and handle contextual errors for a line of text.

    Args:
        line_text (str): The line of text.

    Returns:
        predicted_line (str): The predicted line based on language modeling.
    """
    # Create a spell checker object
    spell = SpellChecker()

    # Tokenize the line text into words
    words = line_text.split()

    # Apply language modeling techniques to the words
    corrected_words = []
    for word in words:
        # Check if the word is misspelled
        if word not in spell:
            # Get the most likely correct spelling
            corrected_word = spell.correction(word)
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)

    # Join the corrected words back into a line
    predicted_line = ' '.join(corrected_words)

    return predicted_line


def save_to_pdf(output_file, recognized_text):
    """
    Saves the recognized text to a PDF file.

    Args:
        output_file (str): The output file path.
        recognized_text (str): The recognized text.

    Returns:
        None
    """
    pdf_writer = PdfWriter()
    pdf_writer.add_page()
    c = canvas.Canvas(output_file)
    c.setFont("Helvetica", 12)
    c.drawString(100, 700, recognized_text)
    c.save()

def main(image_path, dataset_directory):
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Collect the dataset of labeled character images
    dataset, labels = collect_dataset(dataset_directory)

    # Check if the dataset was collected successfully
    if dataset and labels:
        # Split the dataset into training and testing subsets
        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

        # Proceed with further processing or training using the split dataset

        # Perform feature extraction on the training dataset
        train_features = extract_features(X_train)

        # Select an appropriate model for character recognition
        model = select_model()

        # Train the selected model using the extracted features and labels
        train_model(train_features, y_train, model)

        # Perform feature extraction on the testing dataset
        test_features = extract_features(X_test)

        # Load the image
        image = cv2.imread(image_path)

        if image is not None:
            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Recognize characters using the trained model
            recognized_characters = recognize_characters(preprocessed_image, model)

            # Group the recognized characters together to reconstruct text lines or words
            grouped_characters = group_characters(recognized_characters, threshold=10)  # Set the threshold as needed

            # Handle the case where no line break is detected
            if not grouped_characters:
                grouped_characters = [recognized_characters]  # Treat the entire input as a single line

            # Apply language modeling techniques to improve recognition accuracy and handle contextual errors
            predicted_words = apply_language_model(grouped_characters)

            recognized_text = " ".join(predicted_words)

            # Save the recognized text to a PDF file
            output_file = "recognized_text.pdf"
            save_to_pdf(output_file, recognized_text)

            logging.info("Recognized text saved to: %s", output_file)

        else:
            logging.error("Failed to load the image.")
    else:
        logging.error("Failed to collect the dataset.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Character Recognition Program")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("dataset_directory", type=str, help="Path to the directory containing labeled character images.")

    args = parser.parse_args()

    # Run the main program
    main(args.image_path, args.dataset_directory)
