from z_forcing_mlp import Z_Forcing_mlp
from z_forcing_RIMs_latent import Z_Forcing_RIMs
from z_forcing import Z_Forcing


import numpy as np
import tensorflow as tf

model = Z_Forcing(200, 1024, 2048, 1024, 256,400,cond_ln=False)
model_RIM = Z_Forcing_RIMs(200, 1024, 2048, 1024, 256,400,cond_ln=False)
model_mlp = Z_Forcing_mlp(200, 1024, 2048, 1024, 256,400,cond_ln=False)


#fwd,bkw,aux,kld= model.call(np.ones((2,28,200)),np.ones((2,28,200)),np.ones((2,28)),(np.ones((2,28, 2048)),np.ones((2,28, 2048)) ))


#print(fwd)
#print(bkw)
#print(aux)
#print(kld)


fwd,bkw,aux,kld= model_mlp.call(np.ones((2,28,200)),np.ones((2,28,200)),np.ones((2,28)) )


print(fwd)
print(bkw)
print(aux)
print(kld)

#fwd,bkw,aux,kld= model_RIM.call(np.ones((2,28,200)),np.ones((2,28,200)),np.ones((2,28)),(np.ones((2,28, 12288)),np.ones((2,28, 12288)) ))

#print(fwd)
#print(bkw)
#print(aux)
#print(kld)