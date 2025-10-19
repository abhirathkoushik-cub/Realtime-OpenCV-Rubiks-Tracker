# Realtime-OpenCV-Rubiks-Tracker
This project is a Real-time Rubik's Cube face and color tracker using OpenCV and accelerating it with CUDA.

# Quick Snippet
<img width="759" height="695" alt="image" src="https://github.com/user-attachments/assets/7f166503-6ccc-4619-a468-e0f371271bf0" />

# Project Features
- Real-time face detection for up to three visible faces of a Rubik's Cube
- GPU-accelerated image processing pipeline using CUDA for low latency
- Identification of individual sticker colors using HSV color space classification
- Multi-threaded architecture to separate frame capture, processing, and display for optimal performance

# Project Requirements
- C++ 17
- OpenCV 4 enabled with CUDA
- NVIDIA Jetson Orin Nano (Optional, but I have used this to run with CUDA)

# Usage
- make all
- ```./rubiks_cube_tracker -f input_sample.mp4``` OR ```./rubiks_cube_tracker -d /dev/video*``` (if a video device is connected)

# Project Contributors
- Aditya Ganesh (adityaganeshcu2k)
- Abhirath Koushik (abhirathkoushik-cub)
