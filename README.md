# Feature Encoder Module

This directory contains the necessary scripts to preprocess raw video data and extract features for the Automatic Depression Detection model. The primary tool used for feature extraction is [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace). The main script (`main.py`) automates the process of running OpenFace on a dataset of videos and consolidating the results.


### File Descriptions

-   **`backbones/`**: This directory is intended to hold any pre-trained model architectures or utility scripts that might be used by the main encoder model.
-   **`dataset.csv`**: A CSV file that acts as a manifest for your dataset. It should contain paths to the video files and any associated metadata, such as labels or participant IDs.
-   **`dataset.py`**: A Python script defining a custom `Dataset` class. This class is responsible for parsing `dataset.csv` and loading data samples for processing or model training.
-   **`main.py`**: The main executable script for this module. Its primary function is to iterate through the videos listed in `dataset.csv`, use OpenFace to extract facial features, and save the processed features into a unified file.
-   **`model.py`**: This file defines the neural network architecture of the feature encoder itself, which will take the preprocessed features as input.

## Prerequisites

Before running the preprocessing script, ensure you have the following installed:

1.  **Python 3.x**
2.  **OpenFace**: You must have a working installation of OpenFace. The `FeatureExtraction` executable should be accessible from your system's PATH, or you must provide its full path in the `main.py` script.
    -   [OpenFace Installation Guide](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Installation)
3.  **Python Libraries**: Install the required packages using pip.

