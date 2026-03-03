# This project implements a Federated Learning based Intrusion Detection System (IDS) for IoT networks using C++.

Federated Learning allows multiple clients to train models locally using their own data while sharing only model parameters with a central server. The server aggregates these updates to create a global intrusion detection model without sharing raw data.

This approach improves data privacy, distributed learning, and scalability in IoT environments.

# Features

Federated Learning architecture

Privacy-preserving distributed training

Intrusion detection for IoT networks

Multiple client simulation

Aggregation using Federated Averaging (FedAvg)

# Datasets Used

This project uses two network intrusion datasets:

CIC Dataset

CTU Dataset

Both datasets are converted to numeric format and stored in the datasets/ directory.

# Installation

Clone the repository:

git clone https://github.com/<your-username>/federated-iot-ids.git

Install dependencies:

pip install -r requirements.txt

# Running the Project
Run the federated server:

./server

Run clients:

./client

Clients train local models and send updates to the server.
The server aggregates updates to generate a global model.

# Technologies Used

C++

Machine Learning concepts

Federated Learning

Network intrusion datasets