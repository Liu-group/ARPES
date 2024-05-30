# ARPESNet
This repository contains the code for the paper \\
"Explainable Machine Learning Identification of Superconductivity from Single-Particle Spectral Functions".
## Abstract
The traditional method of identifying symmetry-breaking phase transitions through the emergence of a single-particle gap encounters significant challenges in quantum materials with strong fluctuations. To address this, we have developed a data-driven approach using a domain-adversarial neural network trained on simulated spectra of cuprates. This model compensates for the scarcity of experimental data -- a significant barrier to the wide deployment of machine learning in physical research -- by leveraging the abundance of theoretically simulated data. When applied to unlabeled experimental spectra, our model successfully distinguishes the true superconducting states from gapped fluctuating states, without the need for fine temperature sampling across the transition. Further, the explanation of our machine learning model reveals crucial role of the Fermi-surface spectral intensity even in gapped states. It paves the way for robust and direct spectroscopic identification of fluctuating orders, particularly in low-dimensional, strongly correlated materials.

## Dependencies
The code was developed and tested on Python 3.10.10 using CUDA 11.4 with \\
the following Python packages installed:
- 'pytorch_cuda11.3_cudnn8.3.2==1.12.1'
- 'torchvision==0.13.1'
- 'scikit-learn==1.2.2'
- 'captum==0.7.0'

## Usage
First download the ARPES data from https://figshare.com/s/762454387fddddd0987a and create a folder for storing the processed ARPES. Then download/clone this repo and run:
'''
python main.py --data_path [processed_APRES_folder_path]
'''
to get the binary classification results from the paper. To get results with other parameters, one can change the settings in utils/parsing.py.