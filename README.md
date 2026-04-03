"see through CODE view for better understanding" 

Project Synapse: AI-Driven V2X Communication System
Advanced Data Compression & Transmission for Automotive Safety
Project Overview
Synapse is an intelligent Vehicle-to-Everything (V2X) framework designed to optimize data transmission between vehicular nodes and infrastructure. By leveraging Efficient Learned Image Compression (ELIC) and Generative models, this system reduces communication latency and bandwidth requirements, ensuring real-time performance for autonomous and connected vehicles

System Architecture:
The project implements a complete Transmitter-Receiver pipeline optimized for edge computing devices like the Raspberry Pi and centralized monitoring stations.

Core Components:-
transmitter.py: Handles the data acquisition from vehicle sensors and performs high-ratio compression using the GVAE architecture before broadcasting over the network.
pre_trained.py: A specialized utility script that manages the initialization of the neural network, loading pre-trained weights from the elic_models/ directory into memory.
final_receiver1.py: The primary sink node logic that listens for incoming data packets, executes the decompression algorithm, and reconstructs the data with minimal loss for situational awareness.

Key Technical Features:

Edge Optimization: Designed to run on resource-constrained hardware while maintaining low-latency processing.

Robust Protocols: Implements socket-based communication logic reflecting industry-standard Automotive Communication principles.

High-Fidelity Reconstruction: Uses advanced AI models to ensure that safety-critical visual data remains accurate after decompression.

Installation & Usage
Prerequisites
Python 3.8+
PyTorch / TensorFlow
OpenCV & NumPy

Running the System
Initialize the Environment:
Ensure your elic_models/ folder is in the root directory.

Start the Receiver:
Run the receiver on the monitoring station (Laptop):

Bash:
python final_receiver1.py

Start the Transmitter:
Run the transmitter on the source node (Raspberry Pi):

Bash:
python transmitter.py

Development Standards:
This project follows a modular Embedded Systems design approach. All scripts are documented to support collaborative development and meet the rigorous standards expected in Automotive Electronics.

NOTE :
To train the model you need a dataset 
After traning the model it Generates a "pth" file 
Then  you need to set up the folder in both Pi and Receiver
create a folder in both pi and receiver (LAPTOP OR A CAR)
each folder contains the generated pth file and the respective receiver program and transmitter program 
Use the same port number to connect the transmitter and receiver .
