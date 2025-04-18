# Lung-Cancer-Prediction

## Workflows
1. Update config.yaml
2. Update params.yaml
3. Update the entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the dvc.yaml
9. app.py

## How to Run?

### Steps

#### Step 1: Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/Himanshusinghdev1/Lung-Cancer-Prediction
cd Lung-Cancer-Prediction
```

#### Step 2: Create and Activate a Conda Environment
Create a new conda environment with Python 3.8 and activate it:

```bash
conda create -n lung python=3.8 -y
conda activate lung
```

#### Step 3: Install Requirements
Install all necessary dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```


## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/Himanshusinghdev1/Lung-Cancer-Prediction.mlflow \
MLFLOW_TRACKING_USERNAME=Himanshusinghdev1 \
MLFLOW_TRACKING_PASSWORD=e663963ebb08452411148b424509dcd751190e6a \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/Himanshusinghdev1/Lung-Cancer-Prediction.mlflow

export MLFLOW_TRACKING_USERNAME=Himanshusinghdev1 

export MLFLOW_TRACKING_PASSWORD=e663963ebb08452411148b424509dcd751190e6a

```

