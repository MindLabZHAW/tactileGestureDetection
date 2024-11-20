# Backend Code Handbook

This is the handbook of our system for reference. It contains the whole pipeline overview and information for every script inside our system like hyperparameter meanings, script interfaces, etc..

[TOC]

## Algorithm Pipeline



## Scripts Description

### /frankaRobot

#### main.py

##### - Overview

This file is the main file for our system's real-time deployment. In this file, we subscribe to the robot data topic, pass them into our contact detector,  publish the detector result, and save them with another saving node as well. 

##### - Hyperparameters

> RT - Related to Training (Should be changed together or malfunction), ID - Independent(Can be changed only in this file)

- window_length[int]: Detect window length (RT)

- dof[int]: Robot degree of freedom, same as joint number (RT)

- features_num[int]: Number of features used in models (RT)

- class_num[int]: Number of target classes in basic touch type classifier (RT)

- Normalization[Bool]: Flag used to control whether to conduct normalization during deployment (RT)

- ------

-  method[str]: chose the method to deploy(ID)

    - 'KNN': KNN methods

    - 'RNN': RNN methods including LSTM, GRU, and Liquid Neural Network

    - 'TCNN': Brute combines time domain data into T-Image and uses CNN for training, matrix dimension: (window_len, dof, feature_num)

    - 'Freq': Frequency methods including different network structures and different input


- type_network[str]: Sub-selector under methods to specify what inner procedure or network structure is(ID)

  - RNN method
    - 'LSTM': Single layer LSTM network structure
    - 'GRU': Single layer GRU network
    - 'FCLTC': Fully connected network with Liquid Time-Constant neurons
    - 'FCCfC': Fully connected network with Closed-Form Continuous-Time neurons
    - ‘NCPLTC’: Neural Circuit Policy Network structure with Liquid Time-Constant neurons
    - 'NCPCfC': Neural Circuit Policy Network structure with Closed-Form Continuous-Time neurons
  - TCNN method
    - '1L3DTCNN': 1 Layer 3D CNN
    - '2L3DTCNN': 2 Layer 3D CNN
  - Frequency methods
    - '2LCNN': 2 Layers 2D CNN
    - '3LCNN': 3 Layers 2D CNN
    - '2L3DCNN': 2 Layers 3D CNN for Spectrogram
    - 'T2L3DCNN': 2 Layers 3D CNN for fake Spectrogram (column as raw time signal without frequency transform)
  
- model_path_relative[str]: combine the main path and the related path of stored models(ID)

  ------

- MultiClassifier[Bool]: Flag used to control whether use only basic touch_type classifier methods or advanced multi classifiers methods

##### - I/O Interface

- No I/O interface for this file

##### - Algorithm Description

- To be finished

#### ImportModel.py

##### - Overview

##### - Hyperparameters

##### - I/O Interface

##### -Algorithm Description



### /AIModels

### /AIModels/MultiClassifier

#### GestureRecord.py

- Hyperparameters

 