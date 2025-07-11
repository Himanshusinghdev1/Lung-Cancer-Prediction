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

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 404629184544.dkr.ecr.eu-north-1.amazonaws.com/lungcancer

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  404629184544.dkr.ecr.eu-north-1.amazonaws.com

    ECR_REPOSITORY_NAME = lungcancer

