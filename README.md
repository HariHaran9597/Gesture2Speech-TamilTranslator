# Gesture2Speech-TamilTranslator

A real-time Sign Language Recognition (SLR) system that translates hand gestures to spoken Tamil using computer vision and neural networks. This project combines state-of-the-art deep learning with practical accessibility features to bridge communication gaps.

## Key Features

- **Real-time Recognition**: Achieves 92% recognition accuracy in gesture detection using VGG16 architecture
- **Optimized Performance**: 200ms inference latency on Raspberry Pi through TensorFlow Lite and ONNX quantization
- **Interactive Web Interface**: Flask-based responsive web application with real-time video streaming
- **Multilingual Support**: Integrated Google Translate API supporting 10+ languages for subtitles
- **Audio Feedback**: Real-time audio output for seamless communication

## Technical Stack

- **Deep Learning**: VGG16, TensorFlow/Keras
- **Computer Vision**: OpenCV for real-time video processing
- **Optimization**: TensorFlow Lite, ONNX quantization
- **Backend**: Flask web framework
- **Frontend**: Responsive UI with video streaming capabilities
- **APIs**: Google Translate API for multilingual support
- **Hardware Support**: Optimized for Raspberry Pi deployment

## Performance Metrics

- Recognition Accuracy: 92%
- Inference Latency: 200ms
- Language Support: 10+ languages
- Platform: Web-based, accessible from any modern browser

## Project Structure

- `datacollection.py`: Script for gathering training data
- `test.py`: Testing and evaluation script
- `keras_model.h5`: Trained model weights
- `mnist.h5`: Additional model data
- `labels.txt`: Class labels for gesture recognition

## Hardware Requirements

- Raspberry Pi (recommended) or any computer with webcam
- Camera module for video input
- Internet connection for translation services

## Future Improvements

- Expand gesture vocabulary
- Implement offline translation capabilities
- Enhanced mobile device support
- Real-time collaborative features

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under standard open source terms.
