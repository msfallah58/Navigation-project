# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
# add project directory to python path to enable relative imports
import os
import sys
import misc.params as params 

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        self.n = params.dim_state #state dimension
        self.dt = params.dt #time-step
        self.q = params.q #noise covariance 
        pass

    def F(self):
        ############
        n = self.n
        F = np.eye(n)
        F[0, 3] = F[1, 4] = F[2, 5] = self.dt 
        return F
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        n = self.n
        dt = self.dt
        q = self.q
        q1 = q/3 * (dt)**3
        q2 = q/2 * (dt)**2
        q3 = q * dt
        Q = np.zeros((n, n))
        Q[0, 0] = Q[1, 1] = Q[2, 2] = q1
        Q[3, 3] = Q[4, 4] = Q[5, 5] = q3
        Q[0, 2] = Q[1, 3] = Q[2, 0] = Q[2, 4] = Q[3, 1] = Q[3, 5] = Q[4, 2] =  Q[5, 3] = q2
        
        return Q
        
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        F = self.F()
        x = F * track.x # state prediction
        P = F * track.P * F.transpose() + self.Q() # covariance prediction
        track.set_x(x)
        track.set_P(P)
      
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        H = meas.sensor.get_H(track.x) # measurement matrix
        gamma = self.gamma(track, meas) # residual
        S = self.S(track, meas, H) # covariance of residual
        K = track.P * H.transpose()*np.linalg.inv(S) # Kalman gain
        x = track.x + K * gamma # state update
        I = np.identity(self.n)
        P = (I - K * H) * track.P # covariance update
        track.set_x(x)
        track.set_P(P)
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        Gamma = meas.z - meas.sensor.get_hx(track.x)# residual
        return Gamma
        ############
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        cov_residual = H * track.P * H.transpose() + meas.R # covariance of residual
        ############

        return cov_residual
        
        ############
        # END student code
        ############ 