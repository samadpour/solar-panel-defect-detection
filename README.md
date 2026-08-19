# ?? Solar Panel Defect Detection

An AI-powered web application for detecting defects in solar panel images using deep learning.

The application uses a fine-tuned **MobileNetV2** convolutional neural network to perform binary image classification:

* **Reference / No Defect**
* **Defect**

Users can upload a solar panel image through a web browser and receive a prediction without installing Python or the machine-learning environment locally.

## Model

The model uses:

* MobileNetV2
* Transfer learning with ImageNet-pretrained weights
* Global Average Pooling
* Dense classification layer
* Dropout regularization
* Binary classification with sigmoid activation
* Image resolution: 224 × 224 pixels
* RGB input
* Pixel scaling: `1/255`

The training pipeline uses 5-fold stratified cross-validation, followed by model selection for deployment.

## Web Application

The Streamlit application performs the following steps:

1. Accepts an uploaded JPG, JPEG, or PNG image.
2. Converts the image to RGB.
3. Resizes it to 224 × 224 pixels.
4. Scales pixel values to the range `[0, 1]`.
5. Passes the image through the trained MobileNetV2 model.
6. Calculates the probability of a defect.
7. Applies the optimized decision threshold.
8. Displays the final prediction and confidence.

## Project Structure

```text
solar-panel-defect-detection/
?
??? app.py
??? best_model.keras
??? threshold.txt
??? requirements.txt
??? README.md
??? .gitignore
```

## Running Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

The application will open in your web browser.

## Deployment

The application can be deployed using Streamlit Community Cloud.

The GitHub repository should contain:

```text
app.py
best_model.keras
threshold.txt
requirements.txt
README.md
.gitignore
```

The training dataset and training results are not required for inference and should not be uploaded to the public application repository.

## Important

The application should be evaluated using images that were not used during model training.

The displayed confidence represents the model's predicted probability according to the trained classifier and should not be interpreted as a guaranteed probability of physical defect.

## Intended Use

This application is intended for research, demonstration, and educational purposes.

It should not be considered a replacement for professional solar-panel inspection or certified industrial inspection systems.