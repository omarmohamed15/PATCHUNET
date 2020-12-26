% Preparing the patches for the PATCHUNET
clc
clear 
close all

% Loading the Data 
% Synthetic Example
load synthetic_example

% preparing the patches where w1 and w2 are the patch size, while the s1z
% and s2z are the number of shift samples between neighbor windows. The
% default value are w1,w2 =48,48, and s1z,s2z=1,1.
dn1 = DataNoisy;
w1 =48;
w2 =48;
s1z =1;
s2z =1;
dn_patch = yc_patch(dn1,1,w1,w2,s1z,s2z);
% It is better to save .mat as -V 7.3 or later because the size of the generated patch is large.
save Input_Patches dn_patch

% obtain the SNR of the noisy data 
fprintf('The SNR of the Noisy data is %0.3f \n',yc_snr(DataClean,DataNoisy))
