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
│       └── [CollectiongDate]-[ContactLinkNum][GestureType]-[RobotPose][GestureDirection][Round]
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

The dataset we collected is stored in the folder [`DATA/`](./DATA), following the structure described above. It was collected by applying three types of gestures (**Single Tap (ST)**, **Push (P)**, and **Grab (G)** in five directions (left, right, front, back, up) on the robot hand (joint 6 + link 7 + end effector).

- **Single Tap** (ST): A brief, impulse-like touch applied to the robot surface
without sustained pressure, such as pat, poke, or slap;
- **Push** (P): A sustained force applied to the robot with a clear direction,
such as push, pull, or lift;
- **Grab** (G): A prolonged contact in which the hand encloses a link of the robot and applies stable pressure from multiple directions, such as pinch or squeeze;

![GestureType](/image/GestureType.jpg)

The sampling frequency was set to 200 Hz. For each gesture–direction pair, we collected two repetitions per round using a digital glove. Since gesture durations varied, to normalized and balanced the data lengths across different gesture classes, in total we collected: 

| Pose   | Single Tap (ST) | Push (P) | Grab (G) |
|--------|-----------------|----------|----------|
| Pose 1 | 4 rounds        | 1 round  | 1 round  |
| Pose 2 | 4 rounds        | 1 round  | 1 round  |
| Pose 3 | 4 rounds        | 1 round  | 1 round  |


## Inputs & Models

In our work, three types of inputs (**Raw-Time(RT)** Stack, **Short-Time Fourier Transform(STFT)** Spectrogram, **Short-Time Transform(STT)** Pseudo-Spectrogram) are constructed, as illustrated below:

![Inputs](/image/Input_BoundNone.png)

> Parameter settings: Joint Number = 7, Feature Number = 4, Detect Window Size = 28, Sliding Window Size = 16

Based on those inputs, we evaluate the performances of the following network structures(including but not limited to those in this repository):

| Model Name | Input         | Layer Num. | Key Hyperparameters                                                                 |
|------------|---------------|------------|--------------------------------------------------------------------------------------|
| STFT2DCNN  | 3D STFT Image | 5          | 3DConv1 (28×3×3) → 3DConv2 (1×3×3) → 3D Pool (1×1×1) → Flatten → FC                 |
| STFT3DCNN  | 3D STFT Image | 5          | 3DConv1 (7×3×3) → 3DConv2 (1×3×3) → 3D Pool (1×1×1) → Flatten → FC                   |
| STT2DCNN   | 3D STT Image  | 5          | 3DConv1 (28×3×3) → 3DConv2 (1×3×3) → 3D Pool (1×1×1) → Flatten → FC                  |
| STT3DCNN   | 3D STT Image  | 5          | 3DConv1 (4×3×3) → 3DConv2 (7×3×3) → 3D Pool (1×1×1) → Flatten → FC                   |
| RT2DCNN    | 3D RT Image   | 4          | 3DConv (28×3×3) → 3D Pool (1×1×1) → Flatten → FC                                     |
| RT3DCNN    | 3D RT Image   | 5          | 3DConv1 (5×3×3) → 3DConv2 (5×3×3) → 3D Pool (1×1×1) → Flatten → FC                   |

> For Convolution Layer, we only mentioned kernel size as paddings are all 0 and strides are all 1

## Results

The real-time deployment results of the proposed models are summarized below, covering two experiments(also illustrated in the header image) :  
1. Training on Pose 1, testing on Pose 1  
2. Training on Poses 1–3, testing on Pose 4 

![ResultTable](/image/ResultTable.png)

> RD = 0 doesn’t mean there is no recovery delay, but as the delay exceeds our threshold(150 ms), they will all be marked as FP

For the evaluation metrics and further analysis, please refer to our paper.

## Citation

```text
Song, Deqing, et al. "Tactile Gesture Recognition with Built-in Joint Sensors for Industrial Robots." arXiv preprint arXiv:2508.12435 (2025).
```

## Acknowledgements

This work was conducted at MINDLab (ZHAW), supported by the Eurostars project
(Grant No. E!3087) titled SmartSenseAI.

## 🎥 Presentation Video (ICAR 2025)

<iframe width="560" height="315"
src="https://www.youtube.com/embed/KNv19u9G6CQ"
frameborder="0" allowfullscreen>
</iframe>

