# Chord Craft: Crafting Harmonics from Audio

## Overview
Chord Craft is a web application that uses a Convolutional Neural Network (CNN) to predict musical chords from audio files. The application is designed to process audio clips and identify chords with high accuracy. It is built using the [Streamlit](https://streamlit.io/) framework for the web interface, [Librosa](https://librosa.org/doc/main/index.html) for audio processing, and [TensorFlow](https://www.tensorflow.org/) for building and training the neural network model.

## Features
- **Chord Prediction**: Upload a single audio clip (WAV or MP3) to predict the chord.
- **Song Analysis**: Upload a full song to identify chords throughout the audio.
- **Interactive Interface**: User-friendly interface to upload files and view predictions.

## Dataset
The model was trained on a custom dataset of audio clips containing chords played on a ukulele. The dataset includes the following chords:
- A
- C
- F

### Training Details
- **Audio Clips**: 2-second audio clips.
- **Split**: 80% training, 20% testing.
- **Model**: Convolutional Neural Network (CNN).
- **Epochs**: 50.
- **Batch Size**: 16.
- **Accuracy**: 95% on the test set.

## How It Works
1. **Audio Processing**: The audio file is loaded and converted into a spectrogram using the Short-Time Fourier Transform (STFT).
2. **Prediction**: The spectrogram is passed through the trained CNN model to predict the chord.
3. **Results**: The predicted chord and its confidence score are displayed.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Rahul-lalwani-learner/Audio-processing-Chord-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Audio-processing-Chord-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Open the application in your browser at `http://localhost:8501`.
3. Upload an audio file to predict chords.

## File Structure
- `app.py`: Main application file.
- `audio_files/`: Directory for storing uploaded audio files.
- `dataset_chords/`: Directory containing the training dataset.
- `models/`: Directory containing the trained model.

## Requirements
- Python 3.7 or higher
- TensorFlow
- Librosa
- Streamlit

## Screenshots
![Waveform](fullsong%20waveform.png)
![Spectrogram](spectrogram.png)

## Acknowledgments
- [Librosa](https://librosa.org/) for audio processing.
- [TensorFlow](https://www.tensorflow.org/) for building the neural network.
- [Streamlit](https://streamlit.io/) for the web interface.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
