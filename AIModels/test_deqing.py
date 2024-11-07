import pickle
from GestureRecord.py import Gesture

with open('user_data\TestU1\TestG1.pickle', 'rb') as file:
    Gesture_load = pickle.load(file)

print(Gesture_load)