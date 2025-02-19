import os

main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
print(f"main_path is {main_path}")
model_path_relative = os.path.join("AIModels/TrainedModels", "2L3DTCNN_02_18_2025_17-29-08Pose123ES.pth")
model_path = os.path.join(main_path, model_path_relative)