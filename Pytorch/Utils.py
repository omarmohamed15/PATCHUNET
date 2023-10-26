import numpy as np
import h5py
import torch
import torch.nn as nn


#' http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        


def yc_patch(A,l1,l2,o1,o2):

    n1,n2=np.shape(A);
    tmp=np.mod(n1-l1,o1)
    if tmp!=0:
        #print(np.shape(A), o1-tmp, n2)
        A=np.concatenate([A,np.zeros((o1-tmp,n2))],axis=0)

    tmp=np.mod(n2-l2,o2);
    if tmp!=0:
        A=np.concatenate([A,np.zeros((A.shape[0],o2-tmp))],axis=-1); 


    N1,N2 = np.shape(A)
    X=[]
    for i1 in range (0,N1-l1+1, o1):
        for i2 in range (0,N2-l2+1,o2):
            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1));
            X.append(tmp);  
    X = np.array(X)
    return X[:,:,0]


def yc_snr(g,f):
    psnr = 20.*np.log10(np.linalg.norm(g)/np.linalg.norm(g-f))
    return psnr


def yc_patch_inv(X1,n1,n2,l1,l2,o1,o2):
    
    tmp1=np.mod(n1-l1,o1)
    tmp2=np.mod(n2-l2,o2)
    if (tmp1!=0) and (tmp2!=0):
        A     = np.zeros((n1+o1-tmp1,n2+o2-tmp2))
        mask  = np.zeros((n1+o1-tmp1,n2+o2-tmp2)) 

    if (tmp1!=0) and (tmp2==0): 
        A   = np.zeros((n1+o1-tmp1,n2))
        mask= np.zeros((n1+o1-tmp1,n2))


    if (tmp1==0) and (tmp2!=0):
        A    = np.zeros((n1,n2+o2-tmp2))   
        mask = np.zeros((n1,n2+o2-tmp2))   


    if (tmp1==0) and (tmp2==0):
        A    = np.zeros((n1,n2))
        mask = np.zeros((n1,n2))

    N1,N2= np.shape(A)
    ids=0
    for i1 in range(0,N1-l1+1,o1):
        for i2 in range(0,N2-l2+1,o2):
            #print(i1,i2)
    #       [i1,i2,ids]
            A[i1:i1+l1,i2:i2+l2]=A[i1:i1+l1,i2:i2+l2]+np.reshape(X1[:,ids],(l1,l2))
            mask[i1:i1+l1,i2:i2+l2]=mask[i1:i1+l1,i2:i2+l2]+ np.ones((l1,l2))
            ids=ids+1


    A=A/mask;  
    A=A[0:n1,0:n2]

    return A


import numpy

def yc_patch3d( A,mode,l1,l2,l3,s1,s2,s3):


    n1,n2,n3=np.shape(A);

    if mode==1: 

        tmp=np.mod(n1-l1,s1);
        print(tmp,0)
        if tmp!=0:
            #A3 = []
            #A3.append( A)
            #A3.append(np.zeros(s1-tmp,n2,n3))
            #A=np.array(A3)
            A = np.concatenate([A,np.zeros(s1-tmp,n2,n3)],axis=0)
        


        tmp=np.mod(n2-l2,s2)
        print(tmp,1)
        if tmp!=0:
            A=np.concatenate([A,np.zeros((A.shape[0],s2-tmp,n3))],axis=1)
        


        tmp=np.mod(n3-l3,s3)
        print(tmp,2)
        if tmp!=0:
            A = np.concatenate([A,np.zeros((np.shape[0],np.shape[1],s3-tmp))],axis=-1)
        

      

        N1,N2,N3=np.shape(A)
        print(A.shape)
        X=[]
        #for i1=0:s1:N1-l1+1
        for i1 in range(0,N1-l1+1,s1):
            #for i2=0:s2:N2-l2+1
            for i2 in range(0,N2-l2+1,s2):
                #for i3=0:s3:N3-l3+1
                for i3 in range(0,N3-l3+1,s3):
                    #print(A[i1:i1+l1,i2:i2+l2,i3:i3+l3].shape)
                    tmp=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3],(l1*l2*l3))
                    X.append(tmp)

    X = np.array(X)               
    return X

def yc_patch3d_inv(X,mode,n1,n2,n3,l1,l2,l3,s1,s2,s3 ):


    if mode==1: #possible for other patching options



        tmp1=np.mod(n1-l1,s1);
        tmp2=np.mod(n2-l2,s2);
        tmp3=np.mod(n3-l3,s3);
        if (tmp1!=0)  and ( tmp2!=0 ) and ( tmp3!=0):
            A=np.zeros((n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3))
            mask=np.zeros((n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3))
        

        if (tmp1!=0 ) and ( tmp2!=0 ) and ( tmp3==0):
            A=zeros(n1+s1-tmp1,n2+s2-tmp2,n3)
            mask=zeros(n1+s1-tmp1,n2+s2-tmp2,n3)
        

        if (tmp1!=0 ) and ( tmp2==0 ) and ( tmp3==0):
            A=zeros(n1+s1-tmp1,n2,n3)
            mask=zeros(n1+s1-tmp1,n2,n3)
        

        if (tmp1==0 ) and ( tmp2!=0 ) and ( tmp3==0):
            A=zeros(n1,n2+s2-tmp2,n3)
            mask=zeros(n1,n2+s2-tmp2,n3)
        

        if (tmp1==0 ) and ( tmp2==0 ) and ( tmp3!=0):
            A=zeros(n1,n2,n3+s3-tmp3)
            mask=zeros(n1,n2,n3+s3-tmp3)
        

        if ( tmp1==0 ) and ( tmp2==0  ) and ( tmp3==0):
            A=np.zeros((n1,n2,n3))
            mask=np.zeros((n1,n2,n3))
        

        N1,N2,N3=np.shape(A);
        id=0;
        #for i1=0:s1:N1-l1+1
        for i1 in range(0,N1-l1+1,s1):
            #for i2=0:s2:N2-l2+1
            for i2 in range(0,N2-l2+1,s2):
                #for i3=0:s3:N3-l3+1
                for i3 in range(0,N3-l3+1,s3):
                    
                    A[i1:i1+l1,i2:i2+l2,i3:i3+l3]=A[i1:i1+l1,i2:i2+l2,i3:i3+l3]+np.reshape(X[:,id],(l1,l2,l3))
                    mask[i1:i1+l1,i2:i2+l2,i3:i3+l3]=mask[i1:i1+l1,i2:i2+l2,i3:i3+l3]+ np.ones((l1,l2,l3))
                    id=id+1

        A=A/mask;

        A=A[0:n1,0:n2,0:n3]
        
        return A
    

    
def yc_patch5d( A,mode,l1,l2,l3,l4,l5,s1,s2,s3,s4,s5):

    n1,n2,n3,n4,n5=np.shape(A);

    if mode==1 :



        tmp=np.mod(n1-l1,s1);
        if tmp!=0:
            ztmp = np.zeros((s1-tmp,n2,n3,n4,n5))
            A = np.concatenate([A,ztmp],axis=0)
            #A=[A;zeros(s1-tmp,n2,n3,n4,n5)];
            print(A.shape)
        

        tmp=np.mod(n2-l2,s2)
        if tmp!=0:
            #A=[A,zeros(size(A,1),s2-tmp,n3,n4,n5)];
            A = np.concatenate([A,np.zeros((A.shape[0],s2-tmp,n3,n4,n5))])
        

        tmp=np.mod(n3-l3,s3);
        if tmp!=0:
            #A=cat(3,A,zeros(size(A,1),size(A,2),s3-tmp,n4,n5))
            A = np.concatenate([A,np.zeros((A.shape[0],A.shape[1],s3-tmp,n4,n5))],axis=2)
        

        tmp=np.mod(n4-l4,s4)
        if tmp!=0:
            #A=cat(4,A,zeros(size(A,1),size(A,2),size(A,3),s4-tmp,n5)
            A = np.concatenate([A,np.zeros((A.shape[0],A.shape[1],A.shape[2],s4-tmp,n5))],axis=3)
            

        tmp=np.mod(n5-l5,s5)
        if tmp!=0:
            #A=cat(5,A,zeros(size(A,1),size(A,2),size(A,3),size(A,4),s5-tmp))
            A = np.concatenate([A,np.zeros((A.shape[0],A.shape[1],A.shape[2],A.shape[3],s5-tmp))],axis=4)
           
        
        print(A.shape)
        N1,N2,N3,N4,N5=A.shape
        X=[]
        #for i1=0:s1:N1-l1+1
        for i1 in range(0,N1-l1+1,s1):
            #for i2=0:s2:N2-l2+1
            for i2 in range(0,N2-l2+1,s2):
                #for i3=0:s3:N3-l3+1
                for i3 in range(0,N3-l3+1,s3):
                    #for i4=1:s4:N4-l4+1
                    for i4 in range(0,N4-l4+1,s4):
                        #for i5=1:s5:N5-l5+1
                        for i5 in range(0,N5-l5+1,s5):
                            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5],(l1*l2*l3*l4*l5))
                            X.append(tmp)
                            
    X = np.array(X)
    return X

def yc_patch5d_inv( X,mode,n1,n2,n3,n4,n5,l1,l2,l3,l4,l5,s1,s2,s3,s4,s5):


    if mode==1:



        tmp1=np.mod(n1-l1,s1);
        tmp2=np.mod(n2-l2,s2);
        tmp3=np.mod(n3-l3,s3);
        tmp4=np.mod(n4-l4,s4);
        tmp5=np.mod(n5-l5,s5);

        if ( tmp1!=0 ) and (  tmp2!=0 ) and (  tmp3!=0 ) and (  tmp4!=0 ) and (  tmp5!=0):
            A=np.zeros((n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3,n4+s4-tmp4,n5+s5-tmp5))
            mask=np.zeros((n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3,n4+s4-tmp4,n5+s5-tmp5))
        

        if ( tmp1!=0 ) and (  tmp2==0 ) and (  tmp3==0 ) and (  tmp4==0 ) and (  tmp5==0):
            A=np.zeros((n1+s1-tmp1,n2,n3,n4,n5))
            mask=np.zeros((n1+s1-tmp1,n2,n3,n4,n5))
        

        if ( tmp1==0 ) and (  tmp2!=0 ) and (  tmp3==0 ) and (  tmp4==0 ) and (  tmp5==0):
            A=np.zeros((n1,n2+s2-tmp2,n3,n4,n5))
            mask=np.zeros((n1,n2+s2-tmp2,n3,n4,n5))
        

        if ( tmp1==0 ) and (  tmp2==0 ) and (  tmp3!=0 ) and (  tmp4==0 ) and (  tmp5==0):
            A=np.zeros((n1,n2,n3+s3-tmp3,n4,n5))
            mask=np.zeros((n1,n2,n3+s3-tmp3,n4,n5))
        

        if ( tmp1==0 ) and (  tmp2==0 ) and (  tmp3==0 ) and (  tmp4!=0 ) and (  tmp5==0):
            A=np.zeros((n1,n2,n3,n4+s4-tmp4,n5))
            mask=np.zeros((n1,n2,n3,n4+s4-tmp4,n5))
        

        if ( tmp1==0 ) and (  tmp2==0 ) and (  tmp3==0 ) and (  tmp4==0 ) and (  tmp5!=0):
            A=np.zeros((n1,n2,n3,n4,n5+s5-tmp5))
            mask=np.zeros((n1,n2,n3,n4,n5+s5-tmp5))
        

        if ( tmp1==0 ) and (  tmp2==0  ) and (  tmp3==0 ) and (  tmp4==0 ) and (  tmp5==0):
            A=np.zeros((n1,n2,n3,n4,n5))
            mask=np.zeros((n1,n2,n3,n4,n5))
        
        #[tmp1,tmp2,tmp3,tmp4,tmp5]
        N1,N2,N3,N4,N5=A.shape
        id=0;
        #for i1=0:s1:N1-l1+1
        for i1 in range(0,N1-l1+1,s1):
            #for i2=0:s2:N2-l2+1
            for i2 in range(0,N2-l2+1,s2):
                #for i3=0:s3:N3-l3+1
                for i3 in range(0,N3-l3+1,s3):
                    #for i4=1:s4:N4-l4+1
                    for i4 in range(0,N4-l4+1,s4):                                                           
                        #for i5=1:s5:N5-l5+1
                        for i5 in range(0,N5-l5+1,s5):
                                                                                
                            A[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]=A[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]+np.reshape(X[:,id],(l1,l2,l3,l4,l5))
                            mask[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]=mask[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]+np.ones((l1,l2,l3,l4,l5))
                            id=id+1
                    
                
            
        

        A=A/mask;

        A=A[0:n1,0:n2,0:n3,0:n4,0:n5]
    

    return A

def Train(netD,dn,epochsP,BT):

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")# To measure timing
    # Define the optimizer
    optim_d = torch.optim.Adam(netD.parameters(),lr=1e-3,betas=(0.5, 0.9), \
                        eps=1e-8, weight_decay=0)
    l1 = 48
    l2 = 48
    o1 = 1
    o2 = 1
    # Patching 
    dataNoise = yc_patch(dn,l1,l2,o1,o2)
    dataNoise = torch.tensor(dataNoise, dtype=torch.float32).to(device)
    lostbest = 1e5000
    netD.train()  # Set model to training mode
    for ji in range(epochsP):
        kk = 0
        lossall = 0
        for P in range(len(dataNoise)//BT):
            if kk+BT>len(dataNoise):
                dd = dataNoise[kk:]
            else:
                dd = dataNoise[kk:kk+BT]
            kk+=BT
            # zero the parameter gradients
            optim_d.zero_grad()
            loss = nn.MSELoss()
            out = netD(dd)
            loss1 = loss(dd,out)
            loss1.backward()    
            optim_d.step()
            lossall+=loss1
        if lostbest>lossall:
            print('Saving Model')
            torch.save(netD, './best_model.pt')
            lostbest = lossall    
            print('Iter =', ji, '-- Loss =', np.round(lossall.cpu().item(),5))

    netD.eval()
    netD = torch.load('./best_model.pt')
    # Run forward pass
    with torch.no_grad():
        out = netD(dataNoise)    
    out = out.cpu().detach().numpy()

    out = out.T
    n1,n2 = dn.shape
    outB = yc_patch_inv(out,n1,n2,l1,l2,o1,o2)
    
    return outB