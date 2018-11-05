# pex_challenge
Pex Machine Learning Technical Challenge: Develop image classification model

### Run the Code
You can use this library to test whether an image is indoors or outdoors. First, close the libaray:

```
git clone https://www.github.com/niniko1997/pex_challenge
cd pex_challenge/analysis_notebooks
```

To get the predictions, run the code as follows
```
python classifier.py path/to/image
```

You will get an output either reading 'indoor' or 'outdoor'.

Note: The first time you run it will be slow, as it is downloading the feature extractor libary from the youtube 8M starter code. 

### Code Design
The neural network trained to predict whether an image is indoor or outdoor was built as follows:

1. download_data.ipynb: contains code that downloads videos from the youtume 8m library. It then extracts frames from the videos and labels the frames as indoor/ outdoor. This code also generates a training and test split used for the neural networl. 
2. model.ipynb: uses the train test split data to train an artificial neural network and evaluates its effectiveness
3. classifier.py: is a command line interface for testing whether an image is indoor or outdoor based on the trained model
4. feature_extraction.py: is code from the youtube 8m starter code that is implemented to extract the features from images input into classifier.py that are used for in the neural network model
5. test_case.ipynb: demonstrates how to run the classifier.py code and tests its output on a sample image