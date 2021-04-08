from z_forcing import Z_Forcing
import numpy as np
import tensorflow as tf

model = Z_Forcing(200, 1024, 2048, 1024, 256,400,cond_ln=False)

print(model)


one,two,three,four= model.call(np.ones((2,28,200)),np.ones((2,28,200)),np.ones((2,28)),(np.ones((2,28, 2048)),np.ones((2,28, 2048)) ))

print(one)
print(two)
print(three)
print(four)