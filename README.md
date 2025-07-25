# Anomaly Detection using Radiofrequency signals (based on Variational Auto-Encoders)

_Detection of liver nodules using 1D signals from ultrasound acquisitions._  
_LBUM/CRCHUM - Université de Montréal_  

**Note  - This code is the basis for the following publication:**  

- Vianna P, Héroux A, Fohlen A, Nguyen BN, Tang A, Cloutier G. Liver Nodule Anomaly Detection Using Ultrasound Radiofrequency Signals and Variational Autoencoders. IEEE Transactions on Medical Imaging, submitted (under review). Status: 2025-07-25.  

## Introduction
In this approach, a model is trained only with “normal” data to learn representative features of the source domain, to subsequently distinguish abnormal findings based on deviations of features learned. For conducting anomaly detection with variational autoencoders (VAE), models are trained to reconstruct the input, using only data from one class, i.e. negative samples. In this manner, VAE models can be particularly interesting for unsupervised detection of abnormal nodules as they allow for approximating the likelihood of a given datapoint with respect to the distribution they were trained on, which means they can reconstruct well the majority class samples (“normal” data) and underperform in reconstructing other samples.

## Usage
**Note  - Data not available in this repository. Code is presented for demonstration purposes.**  
To explore, please refer to the Jupyter Notebook included in the repository. The code is intended to be easily adaptable: changes to model architecture, hyperparameters, or other configuration settings can be made directly on it.  

The requirements.txt file was generated automatically from a dummy project's virtual environment. You may need to adjust specific package versions depending on your system configuration.  

## Contact information
For any questions or comments, please contact the project authors at:

Pedro Vianna: **pedro.vianna@umontreal.ca**  
Guy Cloutier: **guy.cloutier@umontreal.ca**  
An Tang: **an.tang@umontreal.ca**

## References
N/A
