# PATCHUNET
PATCHUNET: A fully unsupervised and highly generalized deep learning approach for random noise suppression, Geophysical Prospecting, 2020.

We develop a deep learning algorithm (PATCHUNET) to suppress random noise and preserve the coherent seismic signal. The input data are divided into several patches, each patch is encoded to extract the meaningful features. Following this, the extracted features are decompressed to retrieve the seismic signal. Skip connections are used between the encoder and decoder parts, allowing the PATCHUNET to extract high‐order features without losing important information. Besides, dropout layers are used as regularization layers. The dropout layers preserve the most meaningful features belonging to the seismic signal and discard the remaining features. The PATCHUNET is an unsupervised approach that does not require prior information about the clean signal. The input patches are divided into 80% for training and 20% for testing. However, it is interesting to find that the PATCHUNET can be trained with only 30% of the input patches with an effective denoising performance.


We develop a fully unsupervised deep learning approach (PATCHUNET) to suppress random noise based on the idea of the deep image prior. First, a patching technique is utilized to divide the input noisy data into several patches for training and testing. Following this, each patch is encoded to extract the meaningful features, allowing the decoder part to retrieve the clean signal and eliminate the random noise. Next, skip connections are used, between encoder and decoder parts, to extract high‐order features. 

First of all, put all the files in the same directory.

Secondly, kindly, run the codes in sequence:

1- Run "Step1_Patching.m" in Matlab, which generates the input patches for the PATCHUNET. 

2- RUN "Step2_PATCHUNET.ipynb" in Jupyter Notebook, which generates the noiseless output patches. 

3- RUN "Step3_UnPatching.m" in Matlab, which generates the denoised output signal.

In Matlab, the parameter "select" is set to be zero in case of synthetic example and one in case of field example.

