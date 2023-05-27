# Character Recognition

This code provides a character recognition system that can recognize characters in an image using a machine learning model. It includes functionalities for dataset collection, preprocessing, feature extraction, model training, character recognition, language modeling, and PDF output.

## Getting Started

### Prerequisites

To run the code and Docker version, you need to have the following dependencies installed:

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

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/CireWire/fuzzy-octo-rotary-phone.git
   ```

2. Install the required dependencies. You can use `pip` to install them:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The code can be used to recognize characters in an image by following these steps:

1. Collect a dataset of labeled character images. The images should be placed in a directory structure where each character class has its own subdirectory.

2. Adjust the preprocessing, feature extraction, and modeling techniques in the code based on your requirements.

3. Run the code with the following command:

   ```bash
   python character_recognition.py <path_to_image> <path_to_dataset_directory>
   ```

   Replace `<path_to_image>` with the path to the input image file and `<path_to_dataset_directory>` with the path to the directory containing the labeled character images.

4. The code will perform character recognition on the input image and save the recognized text to a PDF file named `recognized_text.pdf`.

## Docker Version

Alternatively, you can use the Docker version to run the character recognition code without worrying about dependencies:

1. Build the Docker image:

   ```bash
   docker build -t character-recognition .
   ```

2. Run the Docker container:

   ```bash
   docker run -v <path_to_host_directory>:/data character-recognition <path_to_image> /data/<dataset_directory>
   ```

   Replace `<path_to_host_directory>` with the path to the directory containing the labeled character images on your host machine. `<dataset_directory>` should be the same as the directory name in the host directory.

3. The Docker container will perform character recognition on the input image and save the recognized text to a PDF file named `recognized_text.pdf` in the current directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
