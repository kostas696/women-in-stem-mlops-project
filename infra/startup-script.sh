#!/bin/bash

# Update system and install core tools
apt-get update
apt-get install -y python3-pip unzip docker.io docker-compose

# Add user to docker group
usermod -aG docker $USER
newgrp docker

# Install ML tooling
pip3 install --upgrade pip
pip3 install apache-airflow mlflow fastapi uvicorn evidently pandas scikit-learn
