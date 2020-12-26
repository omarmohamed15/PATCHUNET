% UnPAtching the Obtained output Patches of the PATHCUNET
clc
clear 
close all

% Loading the Data and the Output Patches of the PATCHUNET
load synthetic_example
load Output_Patches
% UnPatching
[n1,n2]=size(DataClean);
w1 =48;
w2 =48;
s1z =1;
s2z =1;
out=yc_patch_inv(out',1,n1,n2,w1,w2,s1z,s2z);


% obtain the SNR of the noisy data 
fprintf('The SNR of the Noisy data is %0.3f \n',yc_snr(DataClean,DataNoisy))

% obtain the SNR of the noisy data 
fprintf('The SNR of the PATCHUNET is %0.3f \n',yc_snr(DataClean,out))

figure;yc_imagesc(DataClean);
title('Clean Data')
figure;yc_imagesc(DataNoisy);
title('Noisy data')
figure;yc_imagesc(DataNoisy-DataClean);
title('Noise Section')
figure;yc_imagesc(out);
title('Denoised Secton (PATHCUNET)')
figure;yc_imagesc(DataNoisy-out);
title('Removed Noise Section (PATHCUNET)')