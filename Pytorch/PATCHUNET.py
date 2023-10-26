import torch 
import torch.nn as nn

class PatchUNET(nn.Module):
    def __init__(self):
        super(PatchUNET, self).__init__()


               
        self.e1 = nn.Linear(48*48, 128)
        self.ae1 = nn.ELU(1) 

        self.e2 = nn.Linear(128, 64)
        self.ae2 = nn.ELU(1) 
        
        self.e3 = nn.Linear(64, 32)
        self.ae3 = nn.ELU(1) 
        
        self.e4 = nn.Linear(32, 16)
        self.ae4 = nn.ELU(1) 
        
        self.e5 = nn.Linear(16, 8)
        self.ae5 = nn.ELU(1) 
        
        self.e6 = nn.Linear(8, 4)
        self.ae6 = nn.ELU(1) 

        
        self.d1 = nn.Linear(4, 4)
        self.de1 = nn.ELU(1) 

        self.d2 = nn.Linear(8, 8)
        self.de2 = nn.ELU(1) 
        
        self.d3 = nn.Linear(16, 16)
        self.de3 = nn.ELU(1) 
        
        self.d4 = nn.Linear(32, 32)
        self.de4 = nn.ELU(1) 
        
        self.d5 = nn.Linear(64, 64)
        self.de5 = nn.ELU(1) 
        
        self.d6 = nn.Linear(128, 128)
        self.de6 = nn.ELU(1) 
        
        self.sec = nn.Linear(256,48*48)
        
    def forward(self, inputs):

        e1 = self.ae1(self.e1(inputs))
        e2 = self.ae2(self.e2(e1))
        e3 = self.ae3(self.e3(e2))
        e4 = self.ae4(self.e4(e3))
        e5 = self.ae5(self.e5(e4))
        e6 = self.ae6(self.e6(e5))
        
        
        d1 = self.de1(self.d1(e6))
        d1 = torch.cat((d1,e6),axis=-1)
        
        #print(d1.size())
        d2 = self.de2(self.d2(d1))
        d2 = torch.cat((d2,e5),axis=-1)
        
        #print(d2.size())
        d3 = self.de3(self.d3(d2))
        d3 = torch.cat((d3,e4),axis=-1)
        
        #print(d3.size())
        d4 = self.de4(self.d4(d3))
        d4 = torch.cat((d4,e3),axis=-1)
        
        #print(d4.size())
        d5 = self.de5(self.d5(d4))
        d5 = torch.cat((d5,e2),axis=-1)
        
        #print(d5.size())        
        d6 = self.de6(self.d6(d5))
        d6 = torch.cat((d6,e1),axis=-1)
        
        #print(d6.size())
        out = self.sec(d6)
        
        
        return out 

                          
    
                          
  