from z_forcing import Z_Forcing
from z_forcing_RIMs_latent import Z_Forcing_RIMs

import numpy as np
import tensorflow as tf

model = Z_Forcing(200, 1024, 2048, 1024, 256,400,cond_ln=False)
model_RIM = Z_Forcing_RIMs(200, 1024, 2048, 1024, 256,400,cond_ln=False)


#fwd,bkw,aux,kld= model.call(np.ones((2,28,200)),np.ones((2,28,200)),np.ones((2,28)),(np.ones((2,28, 2048)),np.ones((2,28, 2048)) ))


#print(fwd)
#print(bkw)
#print(aux)
#print(kld)

fwd,bkw,aux,kld= model_RIM.call(np.ones((2,28,200)),np.ones((2,28,200)),np.ones((2,28)),(np.ones((2,28, 12288)),np.ones((2,28, 12288)) ))

print(fwd)
print(bkw)
print(aux)
print(kld)