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

The dataset we collected is stored in the folder [`DATA/`](./DATA), following the structure described above. It was collected by applying four types of gestures (**Single Tap (ST)**, **Push (P)**, **Grab (G)**, and an additional **Double Tap (DT)** for extended research) in five directions (left, right, front, back, up) on the robot hand (joint 6 + link 7 + end effector).

- **Single Tap** (ST): A brief, impulse-like touch applied to the robot surface
without sustained pressure, such as pat, poke, or slap;
- **Push** (P): A sustained force applied to the robot with a clear direction,
such as push, pull, or lift;
- **Grab** (G): A prolonged contact in which the hand encloses a link of the robot and applies stable pressure from multiple directions, such as pinch or squeeze;
- **Double Tap** (DT): Two brief, impulse-like touches (2 STs) applied **consecutively** to the robot surface without sustained pressure.

The sampling frequency was set to 200 Hz. For each gesture–direction pair, we collected two repetitions per round using a digital glove. Since gesture durations varied, to normalized and balanced the data lengths across different gesture classes, in total we collected: 

| Pose   | Single Tap (ST) | Push (P) | Grab (G) | Double Tap (DT)|
|--------|-----------------|----------|----------|----------------|
| Pose 1 | 4 rounds        | 1 round  | 1 round  | 2 rounds       |
| Pose 2 | 4 rounds        | 1 round  | 1 round  | 2 rounds       |
| Pose 3 | 4 rounds        | 1 round  | 1 round  | 2 rounds       |

Additional redundant rounds were collected for Pose 1 to support further research and validation.


## Models


## Results

## Citation

```text
Song, Deqing, et al. "Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots." arXiv preprint arXiv:2508.12435 (2025).
```

## Acknowledgements

This work was conducted at MINDLab (ZHAW), supported by the Eurostars project
(Grant No. E!3087) titled SmartSenseAI.