# -----------------------------------------------------------------------------
# FILE NAME:
# USAGE:             python3
# NOTES:             Run
#
# MODIFICATION HISTORY:
# Author             Date           Modification(s)
# ----------------   -----------    ---------------
# Andy Alarcon
# -----------------------------------------------------------------------------

import numpy.matlib as m
import numpy as np
import math


def main():
    testKF = KFReading(1,-1.9512e-65,0,0.001225,8.82147e-199,2.96439e-322,-1.89933e-65,0,0,0)
    print(testKF.matrix_A)
    print(testKF.matrix_initalX)
    print(np.dot(testKF.matrix_A, testKF.matrix_initalX))
    print("\n\n")
    print(testKF.matrix_QNoise)
    print(testKF.matrix_TransA)
    
    

class KFReading:
    def __init__(self, time, odo_x, odo_y, odo_o, imu_o, imu_cov, gps_x, gps_y, gps_covX, gps_covY):
        self.time = time
        self.odo_x = odo_x
        self.odo_y = odo_y
        self.odo_o = odo_o
        self.imu_o = imu_o
        self.imu_cov = imu_cov
        self.gps_x = gps_x
        self.gps_y = gps_y
        self.gps_covX = gps_covX
        self.gps_covY = gps_covY

        #additional matrices
        self.matrix_PredX = np.identity(5)
        self.matrix_CurrX = np.identity(5)
        self.matrix_PredP = np.identity(5)
        self.matrix_K = np.identity(5)

        #Predicted state equation
        #----------------------------------------------------------------------------------------------
        #Create Matrix A
        cos_theta = np.around(np.cos(self.odo_o), decimals=5)
        sin_theta = np.around(np.sin(self.odo_o), decimals=5)
        A_mat = m.matrix([[1, 0, self.time*cos_theta, 0, 0],
                            [0, 1, self.time*sin_theta, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, self.time],
                            [0, 0, 0, 0, 1]])
        self.matrix_A = A_mat


        #Create inital Xstate matrx
        self.velocity = 0.44
        tan_theta = np.around(np.tan(self.odo_o), decimals=5)
        self.wvelocity = self.velocity * (tan_theta / 1)
        X_mat = m.matrix([[self.odo_x], [self.odo_y], [self.velocity], [self.odo_o], [self.wvelocity]])
        self.matrix_initalX = X_mat

        #Predicted proccess equation
        #----------------------------------------------------------------------------------------------

        #Create initial proccess noise covariance matrix
        QN_mat = m.matrix([[.00004, 0, 0, 0, 0],
                            [0, .00004, 0, 0, 0],
                            [0, 0, .0001, 0, 0],
                            [0, 0, 0, .0001, 0],
                            [0, 0, 0, 0, .0001]])
        self.matrix_QNoise = QN_mat

        #Create transpose of A matrix
        self.matrix_TransA = np.transpose(self.matrix_A)

        #Inital covariance matrix P
        PIN_mat = m.matrix([[.01, 0, 0, 0, 0],
                            [0, .01, 0, 0, 0],
                            [0, 0, .01, 0, 0],
                            [0, 0, 0, .01, 0],
                            [0, 0, 0, 0, .01]])
        self.matrix_initalP = PIN_mat

        #Kalman Gain equation
        #--------------------------------------------------------------------------------------------
        self.matrix_H = np.identity(5)
        self.matrix_TransH = np.transpose(self.matrix_H)

        #Measurement Error covariance matrix R
        R_mat = m.matrix([[self.gps_covX, 0, 0, 0, 0],
                            [0, self.gps_covY, 0, 0, 0],
                            [0, 0, .01, 0, 0],
                            [0, 0, 0, self.imu_cov, 0],
                            [0, 0, 0, 0, .01]])
        self.matrix_R = R_mat

        #Inital measurement error covariance matrix R
        IR_mat = m.matrix([[.04, 0, 0, 0, 0],
                            [0, .04, 0, 0, 0],
                            [0, 0, .01, 0, 0],
                            [0, 0, 0, .01, 0],
                            [0, 0, 0, 0, .01]])
        self.matrix_initalR = IR_mat

        #Correction Stage
        #--------------------------------------------------------------------------------------------
        
        #Sensor measurements        
        Z_mat = m.matrix([[self.gps_x], [self.gps_covY], [self.velocity], [self.imu_o], [self.wvelocity]])
        self.matrix_Z = Z_mat



if __name__ == "__main__":
    main()