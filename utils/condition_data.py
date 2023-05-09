import numpy as np
import os

white_image_feature = np.load(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "white_image_feature.npy"))

an_object_feature = np.load(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "an_object_vitl14.npy"))


text_features = {
    "airplane": np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "02691156.npy")),
    "car": np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "02958343.npy")),
    "chair": np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "03001627.npy")),
    "rifle": np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "04090263.npy")),
    "table": np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "04379243.npy"))
}
