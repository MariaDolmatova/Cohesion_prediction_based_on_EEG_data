# Cohesion Prediction Based on EEG data
**By Li and Dolmatova**

## Introduction  
This project aims to classify dyads as high or low cohesive based on EEG synchrony data collected from a social drumming experiment. The study investigates the relationship between neural activity synchrony and interpersonal cohesion using machine learning models.  

---

## Methods  

### EEG Synchrony  
- **Data Preprocessing**:  
  - Sampling rate: 500 Hz.  
  - Tools: MATLAB for signal processing; Python for data reshaping.  
  - Band separation: 5 frequency bands using Butterworth filter.  
  - Segmentation: Time windows of 2s, 4s, 24s, 48s. 
  - Quality Control: Data from 43 pairs of participants retained after artifact detection.


### Cohesion Scores  
- Cohesion assessed using post-session questionnaires (Likert scale: 1-6).  
- Binary classification:  
  - Low cohesion: ≤ 4.5.  
  - High cohesion: > 4.5.

### PCA 
- Test Dataset: smallest (no dissection) and biggest dataset (120 parts).
- Relevant Components: Gamma band, Electrode 7 (C4), electrode 5 (CZ) and electrode 8 (F4).

### Machine Learning Models  

#### SVM Model  
- Dataset: All 5 datasets (2 seconds, 4 seconds, 24 seconds, 48 seconds, 60 seconds time windows and full 4 mins).  
- Techniques: GridSearchCV for hyperparameter tuning, 5-fold cross-validation, weight balance.

#### CNN Model  
- Dataset: 2-second window segments (120 parts).
- Architecture: 1D CNN with 2 convolutional layers.  
- Techniques: Dropout, batch normalization, early stopping, max pooling.  
- Hyperparameters:  
  - Batch size: 16.  
  - Learning rate: 0.001.  
  - Max Epochs: 50.  
  - Loss function: Binary Cross Entropy with Logits Loss.  
  - Optimizer: Adam.  
  - Padding: 1

---

## Results  

### SVM Model  
- Best F1 Score: 0.77

### CNN Model  
- Average F1 Score =  0.55
- Overfitting 

---

## Discussion  

### Key Challenges  
- Small dataset led to overfitting  
- Imbalance in cohesion labels biased model predictions.  

### PCA Analysis Findings  
- Insights align with studies on motor cortex and sensorimotor integration in social cohesion.

### Future Directions  
- Increase dataset size with more participants.  
- Refine synchrony calculation methods.  
- Focus on PFC, ACC regions, and gamma-band activity in future analyses.

---

### Project Data
- All data files should be stored in the `data/` directory

### Project Workflow

1. **Load Data**: Retrieve the required data files from the `data/` directory.
2. **Generate Labels**: Compute cohesion labels from the dataset.
3. **Prepare Input Data**: Process the raw .csv file to create a properly formatted input table.
4. **Visualize Labels**: Generate visualizations to inspect the label distribution.
5. **Model Training**: Train the machine learning model using the processed data.
6. **Result Visualization**: Plot the model's predictions and performance.
7. **Logging**: Maintain logs throughout the process.

### To run the project follow this commands:
Make sure you are in the **project root directory** before running any command.

```bash 
#install Virtual environment
pip install virtualenv
#create virtual environment (serve only this project):
python -m venv venv
#activate virtual environment
.\venv\Scripts\activate  #source venv/bin/activate MAC
+ (venv) should appear as prefix to all command (run next command just after activating venv)
#update venv's python package-installer (pip) to its latest version
pip install --upgrade pip
#install projects packages
pip install -e '.[dev]'   

python main.py
``` 