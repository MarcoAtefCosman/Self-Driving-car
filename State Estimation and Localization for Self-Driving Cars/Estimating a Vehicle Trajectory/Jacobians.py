#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sp
from sympy import *
#calculate the jacobians:
#1)the measurment model with d=0
X_l,Y_l,X_k,Y_k,theta_k=symbols('X_l Y_l X_k Y_k theta_k')
sp.init_printing(use_latex=True)
h=sp.Matrix([[sp.sqrt((X_l-X_k)**2+(Y_l-Y_k)**2)],[atan2(Y_l-Y_k,X_l-X_k)-theta_k]])
h
state=sp.Matrix([X_k,Y_k,theta_k])
H=h.jacobian(state)
H


# In[18]:


######################################3


# In[26]:


import sympy as sp
from sympy import *
x,y,theta,theta_k1,t,v_k,omega_k=symbols('x y theta theta_k-1 t v_k omega_k')
sp.init_printing(use_latex=True)
X=sp.Matrix([[x],[y],[theta]])+t*((sp.Matrix([[cos(theta),0],[sin(theta),0],[0,1]]))@(sp.Matrix([[v_k],[omega_k]])))
X
state=sp.Matrix([x,y,theta])
x_d=X.jacobian(state)
x_d


# In[ ]:




