%Created by Omar M. Saad
%26-12-2020
% UnPAtching the Obtained output Patches of the PATHCUNET
clc
clear 
close all

% Loading the Data and the Output Patches of the PATCHUNET
% Synthetic or Field Example, zero for synthetic and one for field example.
select =1;

switch select 
    case 0 
    load synthetic_example  
    figure;yc_imagesc(DataClean);
    title('Clean Data')
    figure;yc_imagesc(DataNoisy-DataClean);
    title('Noise Section')
    case 1
    load field_example  
end

load Output_Patches
% UnPatching
[n1,n2]=size(DataNoisy);
w1 =48;
w2 =48;
s1z =1;
s2z =1;
out=yc_patch_inv(out',1,n1,n2,w1,w2,s1z,s2z);

switch select
    case 0
    % obtain the SNR of the noisy data 
    fprintf('The SNR of the Noisy data is %0.3f \n',yc_snr(DataClean,DataNoisy))
    % obtain the SNR of the denoised signal corresponding to the PATCHUNET. 
    fprintf('The SNR of the PATCHUNET is %0.3f \n',yc_snr(DataClean,out))
end


figure;yc_imagesc(DataNoisy);
title('Noisy data')
figure;yc_imagesc(out);
title('Denoised Signal (PATHCUNET)')
figure;yc_imagesc(DataNoisy-out);
title('Removed Noise Section (PATHCUNET)')
