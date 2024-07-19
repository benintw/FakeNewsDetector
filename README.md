# Fake News Detector

This project is a machine learning application to detect fake news using a BiLSTM (Bidirectional Long Short-Term Memory) model. The dataset consists of true and fake news articles, and the model is trained to classify the articles as either fake or true.

## Project Structure

```plantext
FakeNewsDetector/
├── config.py
├── data_processing.py
├── dataset.py
├── model.py
├── train_eval.py
├── predict.py
├── visualize.py
└── main.py
```

config.py: Contains the configuration settings for the project.
data_processing.py: Contains functions for processing the data.
dataset.py: Defines the custom dataset class.
model.py: Defines the BiLSTM model.
train_eval.py: Contains functions for training and evaluating the model.
predict.py: Contains the function for making predictions with the trained model.
visualize.py: Contains functions for visualizing training results.
main.py: The main script that ties everything together, trains the model, evaluates it, and plots the results.

## Requirements

Python 3.7 or higher
Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset consists of two CSV files: True.csv and Fake.csv. Each file contains news articles with columns for the title and text of the article. The True.csv file contains legitimate news articles, and the Fake.csv file contains fake news articles.

## Configuration

Configuration settings are stored in the CONFIG class in config.py. These settings include file paths, device settings, batch size, number of epochs, and learning rate.

## How to Run

Prepare the Dataset: Place the True.csv and Fake.csv files in the project directory.

## Train the Model:

Run the main.py script to train the model.

```bash
python main.py
```

## Evaluate the Model:

The model will be evaluated on the validation set during training. The best model will be saved as best_fake_ig_detector.pth.

## Visualize Results:

After training, the script will plot the training and validation losses and accuracies.

## Predict and Evaluate on Test Set:

The script will load the best model and evaluate it on the test set, printing a classification report and plotting a confusion matrix.

## Example Usage

Here is a basic example of how to use the main script:

```bash
python main.py
```

This will train the model, evaluate it, save the best model, and plot the results.
