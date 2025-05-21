<h1 align="center">🧑‍💼 Person Detection using YOLOv8</h1>

<p align="center">
    <em>Real-time person detection for smarter retail security</em>
</p>

--- Person Detection using YOLOv8

This project implements a person detection system using the YOLOv8 model, aimed at identifying potential shoplifting activities in retail environments.

## Features

- Real-time person detection in video streams or images
- Utilizes the state-of-the-art YOLOv8 object detection model
- Easy integration with surveillance camera feeds
- Outputs bounding boxes and confidence scores for detected persons

## Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy

Install dependencies:
```bash
pip install ultralytics opencv-python numpy
```

## Usage

1. Clone the repository.
2. Run the detection script:
    ```bash
    python detect.py --source path/to/video_or_image
    ```
3. View the output with detected persons highlighted.

## Project Structure

- `detect.py` - Main script for running person detection
- `README.md` - Project documentation
- `requirements.txt` - List of dependencies

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model

## License

This project is licensed under the MIT License.
