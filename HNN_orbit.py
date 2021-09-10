import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

import scipy.integrate
solve_ivp  = scipy.integrate.solve_ivp
###########################
# Generate Data
###########################

def HH_evolution(y,t,L):
    
    dydt = np.zeros(4)
    dydt[0]=y[2]
    dydt[1]=y[3]
    dydt[2]=-(y[0]+2*L*y[0]*y[1]);
    dydt[3]=-(y[1]+L*y[0]**2-L*y[1]**2);
    
    return dydt

M=1000
t = np.linspace(0, 100, M+1)
dt=t[2]-t[1]
E=0.166

model_dict = {}
for s0 in range(1,51):

    N1=4 # Length of Training l
    N2=7 # How many point we choose for each
    
    N2_tot=N1*N2
    state=np.zeros([N2_tot*M,5])
    dstate=np.zeros([N2_tot*M,5])
    L_choose=np.array([0.2, 0.4, 0.6, 0.8])

    s=0
    for i in range(0,N1):
    
        L=L_choose[i]
        o=0
        IC=np.zeros([N2,4])
        while o < N2:
        
            q1=(random.random()-0.5)*2
            q2=(random.random()-0.5)*2
            p1=(random.random()-0.5)*2
            p2=(random.random()-0.5)*2
            E_test=1/2*(p1**2+p2**2)+1/2*(q1**2+q2**2)+L*(q1**2*q2-1/3*q2**3)
            V_test=1/2*(q1**2+q2**2)+L*(q1**2*q2-1/3*q2**3)
            if E_test<E/L**2 and V_test>0:
                IC[o,0]=q1;
                IC[o,1]=q2;
                IC[o,2]=p1;
                IC[o,3]=p2;
                o=o+1

        for j in range(0,N2):
    
            sol = odeint(HH_evolution, IC[j,:], t, args=(L,))
    
            state[s*M:(s+1)*M,0]=sol[0:M,0]
            state[s*M:(s+1)*M,1]=sol[0:M,1]
            state[s*M:(s+1)*M,2]=sol[0:M,2]
            state[s*M:(s+1)*M,3]=sol[0:M,3]
            state[s*M:(s+1)*M,4]=L
    
            for k in range(0,M):
                dstate[s*M+k,0:4]=(sol[k+1,:]-sol[k,:])/dt
        
            s=s+1
            
    import os
    cwd=os.getcwd()
    import scipy.io

    scipy.io.savemat(cwd+'/state.mat', mdict={'state': state})        
###########################
# Train HNN
###########################
        
    class HNN(keras.Model):
        def __init__(self, input_dim=4):
            super(HNN, self).__init__()
            self.d1 = tf.keras.layers.Dense(200, activation='tanh')
            self.d2 = tf.keras.layers.Dense(200, activation='tanh')
            self.d3 = tf.keras.layers.Dense(1)
            M = np.zeros([5,5])
            M[0,2]=-1
            M[1,3]=-1
            M[2,0]=1
            M[3,1]=1
            self.M = tf.constant(M, dtype='double')

        def call(self, x):
          y = self.d1(x)
          y = self.d2(y)
          y = self.d3(y)
          return y

        def forward(self, x):
          with tf.GradientTape() as tape:
              y = self.d1(x)
              y = self.d2(y)
              y = self.d3(y)
          y = tape.gradient(y, x)
          y = y @ self.M
          return y

    def train_HNN(state,dstate, learning_rate = 1e-3, epochs = 200):
        model = HNN(input_dim=state.shape[1])
        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for i in range(epochs):
            with tf.GradientTape() as tape1:
                tape1.watch(model.trainable_variables)
                predictions = model.forward(tf.Variable(tf.stack(state)))
                loss = loss_object(tf.Variable(tf.stack(dstate)), predictions)
            gradients = tape1.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print (s0,i, loss)
        return model


    model_dict[s0] = train_HNN(state, dstate, epochs=500, learning_rate=0.005)


def integrate_model_HNN(model_dict, t_span, y0, **kwargs):
    def fun(t, np_x):
        np_x = tf.Variable(tf.reshape(np_x, (1,5)), dtype='double')
        
        dx=np.zeros([5])
        for s0 in range(1,51):
            model = model_dict[s0]
            dx = dx+model.forward(np_x)

        dx=dx/50
        return dx
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

import os
cwd=os.getcwd()
import scipy.io

##
t_span = [0,200]
y0 = np.array([0,0,0.408,0.408,0.5])
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 20000), 'rtol': 1e-12}
hnn_ivp = integrate_model_HNN(model_dict, t_span, y0, **kwargs)
y_pred=hnn_ivp['y']
t_pred=hnn_ivp['t']
scipy.io.savemat(cwd+'/t_pred.mat', mdict={'t_pred': t_pred})
scipy.io.savemat(cwd+'/y_pred_05.mat', mdict={'y_pred': y_pred})

##
t_span = [0,200]
y0 = np.array([0,0,0.408,0.408,0.6])
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 20000), 'rtol': 1e-12}
hnn_ivp = integrate_model_HNN(model_dict, t_span, y0, **kwargs)
y_pred=hnn_ivp['y']
t_pred=hnn_ivp['t']
scipy.io.savemat(cwd+'/t_pred.mat', mdict={'t_pred': t_pred})
scipy.io.savemat(cwd+'/y_pred_06.mat', mdict={'y_pred': y_pred})
  

  
t_span = [0,200]
y0 = np.array([0,0,0.408,0.408,0.7])
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 20000), 'rtol': 1e-12}
hnn_ivp = integrate_model_HNN(model_dict, t_span, y0, **kwargs)
y_pred=hnn_ivp['y']
t_pred=hnn_ivp['t']
scipy.io.savemat(cwd+'/t_pred.mat', mdict={'t_pred': t_pred})
scipy.io.savemat(cwd+'/y_pred_07.mat', mdict={'y_pred': y_pred})
