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

NE=10
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
for s0 in range(0,NE):

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
        for s0 in range(0,NE):
            model = model_dict[s0]
            dx = dx+model.forward(np_x)

        dx=dx/NE
        return dx
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)

def Statistical_HNN(model_dict, t_pred, y_pred, **kwargs):
    
    dt=t_pred[1]-t_pred[0]
    Nt=len(t_pred)
    delta =np.float64(0.001)
    
    w1=np.array([1,0,0,0])
    w2=np.array([0,1,0,0])
    
    d_pred=np.zeros(Nt)
    e=np.zeros(4)
    lya_pred=np.zeros([4,Nt])
    M=np.eye(4)
    Y=np.eye(4)
    for i in range(0,Nt):
        
        
        x0=np.copy(y_pred[:,i])
        x1=np.copy(x0)
        x2=np.copy(x0)
        x3=np.copy(x0)
        x4=np.copy(x0)
        
        x1[0]=x1[0]+delta
        x2[1]=x2[1]+delta
        x3[2]=x3[2]+delta
        x4[3]=x4[3]+delta
        
        dx0=np.zeros([5])
        dx1=np.zeros([5])
        dx2=np.zeros([5])
        dx3=np.zeros([5])
        dx4=np.zeros([5])
        for s0 in range(0,NE):
            model = model_dict[s0]            
            dx0 = dx0+model.forward(tf.Variable([[x0[0],x0[1],x0[2],x0[3],x0[4]]], dtype='double'))
            dx1 = dx1+model.forward(tf.Variable([[x1[0],x1[1],x1[2],x1[3],x1[4]]], dtype='double'))
            dx2 = dx2+model.forward(tf.Variable([[x2[0],x2[1],x2[2],x2[3],x2[4]]], dtype='double'))
            dx3 = dx3+model.forward(tf.Variable([[x3[0],x3[1],x3[2],x3[3],x3[4]]], dtype='double'))
            dx4 = dx4+model.forward(tf.Variable([[x4[0],x4[1],x4[2],x4[3],x4[4]]], dtype='double'))
        dx0=dx0/NE
        dx1=dx1/NE
        dx2=dx2/NE
        dx3=dx3/NE
        dx4=dx4/NE
        
        J=np.zeros([4,4])
        for j in range(0,4):
            J[0,j]=(dx1[0,j]-dx0[0,j])/delta
            J[1,j]=(dx2[0,j]-dx0[0,j])/delta
            J[2,j]=(dx3[0,j]-dx0[0,j])/delta
            J[3,j]=(dx4[0,j]-dx0[0,j])/delta
        
        J=np.eye(4)+J*dt
        M=J @ M
        w1t=M @ w1
        w2t=M @ w2
        
        w1t=w1t/np.sqrt(np.sum(w1t**2))
        w2t=w2t/np.sqrt(np.sum(w2t**2))
        
        d1=np.sqrt(np.sum((w1t-w2t)**2));
        d2=np.sqrt(np.sum((w1t+w2t)**2));
        d_pred[i]=np.min([d1,d2])
        
        Y=J @ Y
        q, r = np.linalg.qr(Y)
        
        lb = np.diag(r);
        Y = q;
        e=e+ np.log(abs(lb));
        lya_pred[:,i]=e/((i+1)*dt)
        
    return d_pred,lya_pred
    

import os
cwd=os.getcwd()
import scipy.io

import os
cwd=os.getcwd()
from scipy.io import loadmat
x = loadmat(cwd+'/Initial_Condition.mat')
IC=x.get('IC')
L_choose=x.get('L_choose')

t_span = [0,10**3]
i=0
for j in range(0,10):
        
    y=np.zeros(5)
    y[0:4]=IC[i,j,:]
    y[4]=L_choose[0,i]
    kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 10**5), 'rtol': 1e-12}
    hnn_ivp = integrate_model_HNN(model_dict, t_span, y, **kwargs)

    t_pred=hnn_ivp['t']
    y_pred=hnn_ivp['y']

    d_pred,lya_pred=Statistical_HNN(model_dict, t_pred, y_pred, **kwargs)
    
    d_pred=d_pred[range(0,10**5,100)]
    lya_pred=lya_pred[:,range(0,10**5,100)]
    
    scipy.io.savemat(cwd+'/data_'+str(i+1)+'_'+str(j+1)+'.mat', mdict={'lya_pred': lya_pred,'d_pred': d_pred})
