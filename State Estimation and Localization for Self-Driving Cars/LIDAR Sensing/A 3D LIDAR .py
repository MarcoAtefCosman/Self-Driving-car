#!/usr/bin/env python
# coding: utf-8

# In[2]:


#A single 3D LIDAR reading consists of elevation, azimuth and range measurements (ϵ,α,r)=(5∘,10∘,4 m) . Assuming that the measurements are noiseless, calculate the position of this point in the Cartesian sensor frame.
#Enter your result in the area below as comma separated list of values, in the (x, y, z)(x,y,z) format, e.g (1.22, 2.33, 3.44).


# In[29]:


#1
from numpy import *
from numpy.linalg import inv
import numpy as np

def sph_to_cart(epsilon, alpha, r):
  """
  Transform sensor readings to Cartesian coordinates in the sensor frames. 
  """
  p = np.zeros(3)  # Position vector 
  # Your code here
  
  p[0] = r * np.cos(epsilon) * np.cos(alpha)
  p[1] = r * np.cos(epsilon) * np.sin(alpha)
  p[2] = r * np.sin(epsilon)
  
  
  return p
if __name__ == '__main__':
  
  epsilon=5*(np.pi)/180
  alpha=10*(np.pi)/180
  r=4
  param_est_result = sph_to_cart(epsilon,alpha,r)
  print(param_est_result)


# In[30]:


#2
from numpy import *
from numpy.linalg import inv
import numpy as np


def estimate_params(P):
  """
  Estimate parameters from sensor readings in the Cartesian frame.
  Each row in the P matrix contains a single measurement.
  """
  
  param_est = zeros(3)
  
  # Your code here
  x=np.shape(P)
  n=x[0]
  A=np.zeros((n,3))
  b=np.zeros((n,1))
  
  for i in range(0,n):
    A[i,0]=1
    A[i,1]=P[i,0]
    A[i,2]=P[i,1]
    b[i,0]=P[i,2]
    
  
    
  x_hat=inv(A.T@A)@A.T@b
  
  param_est[0]=x_hat[0,0]
  param_est[1]=x_hat[1,0]
  param_est[2]=x_hat[2,0]
  
  return param_est 

if __name__ == '__main__':
  P=np.array([[50,31,18],[12,17,42]])
  x_hat=estimate_params(P)
  print(x_hat)


# In[ ]:




