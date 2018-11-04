import numpy as np
import pandas as pd
from pathlib import Path
import os
import math
import sklearn
from glob import glob
import tensorflow as tf
from IPython.display import YouTubeVideo
from feature_extraction import YouTube8MFeatureExtractor
from PIL import Image

def load_model():
    # load the model from disk
    import pickle 
    filename = ('../model.sav')
    loaded_model = pickle.load(open(filename, 'rb'))
    
    return loaded_model

def mod_features(features):
    X = pd.DataFrame([])
    row_dict = {}
    for i in range(len(features)):
        row_dict['rgb' + str(i)] = features[i]

    X = X.append(row_dict, ignore_index=True)
    return X

def main():
    import argparse
    import sys
    from PIL import Image

    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(
        description="Image classifier as indoor or outdoor.",
        fromfile_prefix_chars='@',
        )
    # Specify parameters
    parser.add_argument(
        "image",
        help="path to the image that you want to classify",
        metavar="image")
    
    # extract parameters from comman line input
    args = parser.parse_args()
    image = vars(args)['image']

    # Instantiate extractor for feature extraction
    extractor = YouTube8MFeatureExtractor()
    image_file = image
    im = np.array(Image.open(image_file))
    features = extractor.extract_rgb_frame_features(im)

    # Turn features into testable format
    X = mod_features(features)

    # load the model
    # load the model from disk
    model = load_model()

    result = model.predict(X)
    if result == 1:
        print("Indoor")
    else:
        print("Outdoor")


if __name__ == '__main__':
    main()