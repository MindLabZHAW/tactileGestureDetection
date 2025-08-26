# Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots

![IntroImage](/image/ExperimentSetup.png)

This repository contains the code, dataset ,and other resources for the paper:
**Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots** (2025)
\> [[Paper Link]](https://arxiv.org/abs/2508.12435)

This project explores deep learning methods, specifically CNN-based approaches, that rely solely on industrial collaborative robots’ (cobots) built-in joint sensors — eliminating the need for external sensors such as vision systems or tactile skins.

## Features

- External-sensors-free gesture recognition solution for cobots
- Multiple input formats, including spectrogram-based and non-spectrogram representations
- Benchmark and evaluation of various 2D/3D CNN architectures
- Open-source dataset for training and testing

## Main Repository Structure

```text
.
├── AIModels/                     # Model Training & Saving Scripts
│   ├── MultiClassifier/          # New approach using multi-classifier for customized gestures (Not yet complete)
│   ├── Freqmodel_withVal.py      # Training & saving script for STFT and STT input CNNs
│   └── TimeCNNmodel_withVal.py   # Training & saving script for RT image input CNNs
├── DATA/                         # Data Storage
│   ├── Labeled_data/             # Folder used to store labeled data during preprocessing
│   ├── rawData                   # Pose 1 raw data storage (ST 4 round + DT 2 round + P & G 1 round)
│   ├── rawData                   # Pose 1 raw redundant data storage (DT & P & G another 1 round)
│   ├── rawData                   # Pose 2 raw data storage (ST 4 round + DT 2 round + P & G 1 round)
│   ├── rawData                   # Pose 3 raw data storage (ST 4 round + DT 2 round + P & G 1 round)
│   ├── rawData                   # Folder used to place USING raw data, adjust when using
│   ├── STFT_images/              # Folder used to store STFT images after preprocessing 
│   └── T_images/                 # Folder used to store STT images after preprocessing 
├── frankaRobot/                  # Real-time Implementation Scripts
│   ├── demo_xxx.py               # 2 Demos with already adjusted hyperparameters
│   ├── ImportModel.py            # Defined Classes and Functions used when importing models
│   └── main.py                   # Main real-time deployment script
├── ProcessData/                  # Data Preprocessing Scripts (Further information in README.md under ProcessData/)
├── Pose4.task                    # Franka desk task file moving robot to pose 4
└── README.md
```

## Installation

For the environment setup please follow the [Contact Interpretation System](https://github.com/MindLabZHAW/contactInterpretation)'s guidance. This repository also includes the scripts collecting and saving raw data from digital gloves.

## Dataset

## Models

## Results

## Citation

```text
Song, Deqing, et al. "Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots." arXiv preprint arXiv:2508.12435 (2025).
```

## Acknowledgements

This work was conducted at MINDLab (ZHAW), supported by the Eurostars project
(Grant No. E!3087) titled SmartSenseAI.