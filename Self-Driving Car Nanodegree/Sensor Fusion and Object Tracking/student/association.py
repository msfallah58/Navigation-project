# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        # the following only works for at most one track and one measurement
        No_tracks = len(track_list)
        No_meas = len(meas_list)
        # initialize unassigned tracks and unassigned measurements lists to be updated in the "associate_and_update" method
        self.unassigned_tracks = list(range(No_tracks)) 
        self.unassigned_meas = list(range(No_meas)) 
        # initialize association matrix
        self.association_matrix = np.inf*np.ones((No_tracks, No_meas))
        for i in range(No_tracks):
            track = track_list[i]
            for j in range(No_meas):
                meas = meas_list[j]
                dist = self.MHD(track,meas,KF)
                if self.gating(dist, meas.sensor):
                    self.association_matrix[i][j] = dist
        return
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        A = self.association_matrix
        if np.min(A) == np.inf:
            return np.nan, np.nan

        # get indices of minimum entry
        ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape) 
        ind_track = ij_min[0]
        ind_meas = ij_min[1]

        # delete row and column for next update
        A = np.delete(A, ind_track, 0) 
        A = np.delete(A, ind_meas, 1)
        self.association_matrix = A

        # update this track with this measurement
        update_track = self.unassigned_tracks[ind_track] 
        update_meas = self.unassigned_meas[ind_meas]

        # remove this track and measurement from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
        
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        if sensor.name == "lidar":
            DOF = 2 # LiDAR DOF, features = (x,y,z)
        elif sensor.name == "camera":
            DOF = 1 # camera DOF,features = (x,y)
        # calculate the Inverse Cumulative Distribution Function (I-CDF) for certain percentile (gating_threshold)
        limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)
        if MHD < limit:
            return True
        else:
            return False    
        
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############ 
        H = meas.sensor.get_H(track.x)
        gamma = meas.z - meas.sensor.get_hx(track.x)
        S = H * track.P * H.transpose() + meas.R
        MHD = gamma.transpose()*np.linalg.inv(S)*gamma # Mahalanobis distance formula
        return MHD
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        if len(meas_list) == 0:
            return
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            if meas_list[0].sensor.name == 'lidar':
                manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        if meas_list[0].sensor.name == 'lidar':
            manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)