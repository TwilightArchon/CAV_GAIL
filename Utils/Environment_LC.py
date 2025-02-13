import os
import numpy as np
import torch
import torch.nn as nn


class IDM(nn.Module):
    def __init__(self, 
        speed_desire, 
        time_gap, 
        acc_max, 
        dec_desire, 
        distance_jam
        ):
        super(IDM, self).__init__()  
        self.v_0 = speed_desire
        self.T = time_gap
        self.a = acc_max
        self.b = dec_desire
        self.s_0 = distance_jam
        
    def forward(self, v_i, v_delta, s_i):
        s_temp = self.s_0 + v_i * self.T + v_i * v_delta / (2 * ((self.a * self.b)**0.5))
        output = self.a * (1 - (v_i / self.v_0)**4 - (s_temp / s_i)**2) 
        
        if output <= -8:  # could not be smaller than emergency braking
            output = -8 
        return output

class MOBIL(nn.Module):
    def __init__(self, 
        altruism_new, 
        altruism_old, 
        b_safe, 
        threshold
        ):
        super(MOBIL, self).__init__()    
        self.p_n = altruism_new
        self.p_o = altruism_old
        self.b_safe = b_safe
        self.threshold = threshold
        
    def forward(self, a_new, a_old, a_n_new, a_n_old, a_o_new, a_o_old):
        output = 0      
        if a_new - a_old + self.p_n * (a_n_new - a_n_old)  +  self.p_o *(a_o_new - a_o_old) > self.threshold:
            output = 1 #change lane
        if a_n_new < self.b_safe:
            output = 0      
        return output

class EIDM(nn.Module):
    def __init__(self):
        super(EIDM, self).__init__()  
   
    def forward(self, spacing, v, v0, cv_arg, s0=2, a0=0.8, b0=1.5, T=1.2, sigma=4, Kf=0.85, Kg=0.6):    
        """
        Return acceleration of CAV
        *cv_args is a tuple of relative speed and acceleration between two CAVs: (relative_speed, relative_acceleration)
        """           
        CAV_count = len(cv_arg)//2
        temp_v, temp_a = 0, 0
        alpha, beta = np.zeros((1,CAV_count)), np.zeros((1,CAV_count))
        if CAV_count > 0:
            for i in range(CAV_count):
                alpha[i] = 0.3/(0.3+np.exp(i))
                beta[i] = 0.25/(0.25+np.exp(i))
                temp_v += alpha[i]*cv_arg[2*i]
                temp_a += beta[i]*cv_arg[2*i+1]
            s_star = s0 + v*T - v*alpha[0]*cv_arg[0]/(2*np.sqrt(a0*b0))
        else:
            print('Error because no CAV speed and acceleration input!')
        a_free = a0*(1-(v/v0)**sigma)
        a_int = -a0*(s_star/spacing)**2
        acc = a_free + a_int
        output = Kf*acc + Kg*(temp_v + temp_a)
        
        if output <= -8:  # could not be smaller than emergency braking
            output = -8 
        return output            

class ENVIRONMENT(nn.Module):  
    def __init__(self, 
        para_B,                                # IDM driver style: aggressive, normal, cautious
        para_A, 
        noise = True                           # stochastic parameters for IDM or deterministic parameters
        ):
        super(ENVIRONMENT, self).__init__()       
        
        self.t              = 0                 # time     
        self.loop_end       = False             # one episode is finished
        self.noise          = noise
        self.lane_change    = 0
        
        self.v_0             = 30                
        self.veh_len         = 5.0
        self.lane_wid        = 3.75
        self.sequence_length = 5
        
        ## Hyperparameter of vehicles##
        self.MOBIL_para = {
            'aggressive' : [0, 0, -8.0, 0],
            'normal'     : [0.05, 0, -5.0, 0],
            'cautious'   : [0.05, 0, -2.0, 0]
        }
        self.IDM_para = {
            'aggressive' : [1.6, 1.05, 1.54],  # T, a, b
            'normal'     : [2.57, 0.87, 1.14],
            'cautious'   : [3.16, 0.8, 1.08],
            '2000'       : [1.6, 0.73, 1.67]
        }
            
        self.para_MOBIL = self.MOBIL_para['normal']
        self.MOBIL_B = MOBIL(self.para_MOBIL[0], self.para_MOBIL[1], self.para_MOBIL[2], self.para_MOBIL[3])
        
        self.para_A  = self.IDM_para[para_A]
        self.para_B  = self.IDM_para[para_B]
        self.para_C  = self.IDM_para['2000']
        self.para_D  = self.IDM_para['2000']
        self.para_G  = self.IDM_para['2000']
        self.para_H  = self.IDM_para['2000']

        if self.noise == True:
            T_B   = np.random.normal(self.para_B[0], 0.2)
            a_B   = np.random.normal(self.para_B[1], 0.08)
            b_B   = np.random.normal(self.para_B[2], 0.08)
            self.s_0_B = np.random.normal(2, 0.5)
            self.IDM_B = IDM(self.v_0, T_B, a_B, b_B, self.s_0_B)
            T_B   = 1.2
            a_B   = 0.8
            b_B   = 1.8
            self.s_0_B = 2
            self.EIDM_B = EIDM()

            T_D   = np.random.normal(self.para_D[0], 0.2)
            a_D   = np.random.normal(self.para_D[1], 0.08)
            b_D   = np.random.normal(self.para_D[2], 0.08)
            s_0_D = np.random.normal(2, 0.5)
            self.IDM_D = IDM(self.v_0, T_D, a_D, b_D, s_0_D)

            T_A   = np.random.normal(self.para_A[0], 0.2)
            a_A   = np.random.normal(self.para_A[1], 0.08)
            b_A   = np.random.normal(self.para_A[2], 0.08)
            self.s_0_A = np.random.normal(2, 0.5)
            self.IDM_A = IDM(self.v_0, T_A, a_A, b_A, self.s_0_A)

            T_C   = np.random.normal(self.para_C[0], 0.2)
            a_C   = np.random.normal(self.para_C[1], 0.08)
            b_C   = np.random.normal(self.para_C[2], 0.08)
            s_0_C = np.random.normal(2, 0.5)
            self.IDM_C = IDM(self.v_0, T_C, a_C, b_C, s_0_C)

            T_G = np.random.normal(self.para_G[0], 0.2)
            a_G = np.random.normal(self.para_G[1], 0.08)
            b_G = np.random.normal(self.para_G[2], 0.08)
            s_0_G = np.random.normal(2, 0.5)
            self.IDM_G = IDM(self.v_0, T_G, a_G, b_G, s_0_G)

            T_H = np.random.normal(self.para_H[0], 0.2)
            a_H = np.random.normal(self.para_H[1], 0.08)
            b_H = np.random.normal(self.para_H[2], 0.08)
            s_0_H = np.random.normal(2, 0.5)
            self.IDM_H = IDM(self.v_0, T_H, a_H, b_H, s_0_H)

        else:
            T_B   = np.random.normal(self.para_B[0], 0)
            a_B   = np.random.normal(self.para_B[1], 0)
            b_B   = np.random.normal(self.para_B[2], 0)
            self.s_0_B = np.random.normal(2, 0)
            self.IDM_B = IDM(self.v_0, T_B, a_B, b_B, self.s_0_B)            
            T_B   = 1.2
            a_B   = 0.8
            b_B   = 1.8
            self.s_0_B = 2
            self.EIDM_B = EIDM()

            T_D   = np.random.normal(self.para_D[0], 0)
            a_D   = np.random.normal(self.para_D[1], 0)
            b_D   = np.random.normal(self.para_D[2], 0)
            s_0_D = np.random.normal(2, 0)
            self.IDM_D = IDM(self.v_0, T_D, a_D, b_D, s_0_D)

            T_A   = np.random.normal(self.para_A[0], 0)
            a_A   = np.random.normal(self.para_A[1], 0)
            b_A   = np.random.normal(self.para_A[2], 0)
            self.s_0_A = np.random.normal(2, 0)
            self.IDM_A = IDM(self.v_0, T_A, a_A, b_A, self.s_0_A)

            T_C   = np.random.normal(self.para_C[0], 0)
            a_C   = np.random.normal(self.para_C[1], 0)
            b_C   = np.random.normal(self.para_C[2], 0)
            s_0_C = np.random.normal(2, 0)
            self.IDM_C = IDM(self.v_0, T_C, a_C, b_C, s_0_C) 

            T_G = np.random.normal(self.para_G[0], 0)
            a_G = np.random.normal(self.para_G[1], 0)
            b_G = np.random.normal(self.para_G[2], 0)
            s_0_G = np.random.normal(2, 0)
            self.IDM_G = IDM(self.v_0, T_G, a_G, b_G, s_0_G)

            T_H = np.random.normal(self.para_H[0], 0)
            a_H = np.random.normal(self.para_H[1], 0)
            b_H = np.random.normal(self.para_H[2], 0)
            s_0_H = np.random.normal(2, 0)
            self.IDM_H = IDM(self.v_0, T_H, a_H, b_H, s_0_H)
        
        ########### Data initialization##########
        # A: target vehicle  lane 1
        # B: ego vehicle     lane 2
        # E: leader          lane 1
        # F: leader          lane 2
        # C: follower        lane 1
        # D: follower        lane 2
        # G: follower to C   lane 1
        # H: follower to D   lane 2
        
        Traj1         = np.loadtxt('Utils/trajectory_1.txt', delimiter='\t')
        Traj2         = np.loadtxt('Utils/trajectory_2.txt', delimiter='\t')
        self.Time_len = 500   # < 1500
        
        self.F      = np.zeros((self.Time_len,3))
        self.F[:,2] = Traj1[300:300+self.Time_len,2] #acc
        self.F[0,1] = 8.0         #speed
        self.F[0,0] = 20          #position
        for i in range (self.Time_len-1):
            self.F[i+1,1] = self.F[i,1] + 0.1 * self.F[i,2]   #speed
            self.F[i+1,0] = self.F[i,0] + 0.1 * self.F[i,1]   #position
            
        self.E      = np.zeros((self.Time_len,3))
        self.E[:,2] = np.concatenate((Traj2[:228,2], Traj2[840:840+self.Time_len-228,2])) #acc
        self.E[0,1] = 8.0    #speed
        self.E[0,0] = 0      #position
        for i in range (self.Time_len-1):
            self.E[i+1,1] = self.E[i,1] + 0.1 * self.E[i+1,2]   #speed
            self.E[i+1,0] = self.E[i,0] + 0.1 * self.E[i+1,1]   #position
        
        self.B = np.zeros((self.Time_len,6))  # longitudinal: 0 - posiiton, 1 - speed, 2 - acc; Lateral: 3 - posiiton, 4 - speed, 5 - acc 
        self.D = np.zeros((self.Time_len,3))
        self.A = np.zeros((self.Time_len,3))
        self.C = np.zeros((self.Time_len,3))
        self.G = np.zeros((self.Time_len,3))
        self.H = np.zeros((self.Time_len,3))
                
        self.B[0,2] = 0                                                                                     #acc
        self.B[0,1] = self.F[0,1]                                                                                #speed
        self.B[0,0] = self.F[0,0] - (self.s_0_B + T_B * self.B[0,1])/((1 - (self.B[0,1] / 30.0)**4)**0.5) - self.veh_len             #position
        self.B[0,5] = 0                                                                                     #acc
        self.B[0,4] = 0                                                                                     #speed
        self.B[0,3] = self.lane_wid * 1.5                                                                        #position

        self.D[0,2] = 0                                                                                     #acc
        self.D[0,1] = self.B[0,1]                                                                                #speed
        self.D[0,0] = self.B[0,0] - (s_0_D + T_D *self.D[0,1])/((1 - (self.D[0,1] / 30.0)**4)**0.5) - self.veh_len             #position

        gap_temp    = 30.0
        self.A[0,2] = 0                                                                                     #acc
        self.A[0,1] = self.E[0,1]                                                                                #speed
        self.A[0,0] = self.E[0,0] - gap_temp - (self.s_0_A + T_A * self.A[0,1])/((1 - (self.A[0,1] / 30.0)**4)**0.5) - self.veh_len  #position            #position

        self.C[0,2] = 0                                                                                     #acc
        self.C[0,1] = self.A[0,1]                                                                                #speed
        self.C[0,0] = self.A[0,0] - (s_0_C + T_C * self.C[0,1])/((1 - (self.C[0,1] / 30.0)**4)**0.5) - self.veh_len             #position

        self.G[0,2] = 0                                                                                     #acc
        self.G[0,1] = self.C[0,1]                                                                                #speed
        self.G[0,0] = self.C[0,0] - (s_0_G + T_G * self.G[0,1])/((1 - (self.G[0,1] / 30.0)**4)**0.5) - self.veh_len             #position 

        self.H[0,2] = 0                                                                                     #acc
        self.H[0,1] = self.G[0,1]                                                                                #speed
        self.H[0,0] = self.G[0,0] - (s_0_H + T_H * self.H[0,1])/((1 - (self.H[0,1] / 30.0)**4)**0.5) - self.veh_len             #position
        
        self.Dat          = np.zeros([self.Time_len,3*8+1+3])      # E, A, C, F, B, D
        self.Dat[:,0:3]   = self.E[:,:]                            # position, speed, acc
        self.Dat[0,3:6]   = self.A[0,:] 
        self.Dat[0,6:9]   = self.C[0,:] 
        self.Dat[:,9:12]  = self.F[:,:] 
        self.Dat[0,12:15] = self.B[0,0:3] 
        self.Dat[0,15:18] = self.D[0,:] 
        self.Dat[0,18:21] = self.G[0,:] 
        self.Dat[0,21:24] = self.H[0,:]
        self.Dat[0,24]    = 0                                       # lane-change decision
        self.Dat[0,25:28] = self.B[0,3:6]                           # lateral position, yaw, yaw rate
    
    def reset(self):       
        self.t              = 0 
        self.loop_end       = False
        
        # Dataset containing all information
        self.Dat          = np.zeros([self.Time_len,3*8+1+3])      # E, A, C, F, B, D
        self.Dat[:,0:3]   = self.E[:,:]                            # longitudinal position, speed, acc
        self.Dat[0,3:6]   = self.A[0,:] 
        self.Dat[0,6:9]   = self.C[0,:] 
        self.Dat[:,9:12]  = self.F[:,:] 
        self.Dat[0,12:15] = self.B[0,0:3] 
        self.Dat[0,15:18] = self.D[0,:] 
        self.Dat[0,18:21] = self.G[0,:] 
        self.Dat[0,21:24] = self.H[0,:]
        self.Dat[0,24]    = 0                                      # lane-change decision
        self.Dat[0,25:28] = self.B[0,3:6]                          # lateral position, yaw angle, yaw rate

    def AB_update(self, para_A, para_B):
        self.para_A  = self.IDM_para[para_A]
        self.para_B  = self.IDM_para[para_B]

        if self.noise == True:
            T_B   = np.random.normal(self.para_B[0], 0.2)
            a_B   = np.random.normal(self.para_B[1], 0.08)
            b_B   = np.random.normal(self.para_B[2], 0.08)
            self.s_0_B = np.random.normal(2, 0.5)
            self.IDM_B = IDM(self.v_0, T_B, a_B, b_B, self.s_0_B)
            
            T_A   = np.random.normal(self.para_A[0], 0.2)
            a_A   = np.random.normal(self.para_A[1], 0.08)
            b_A   = np.random.normal(self.para_A[2], 0.08)
            self.s_0_A = np.random.normal(2, 0.5)
            self.IDM_A = IDM(self.v_0, T_A, a_A, b_A, self.s_0_A)
        else:
            T_B   = np.random.normal(self.para_B[0], 0)
            a_B   = np.random.normal(self.para_B[1], 0)
            b_B   = np.random.normal(self.para_B[2], 0)
            self.s_0_B = np.random.normal(2, 0)
            self.IDM_B = IDM(self.v_0, T_B, a_B, b_B, self.s_0_B)
            
            T_A   = np.random.normal(self.para_A[0], 0)
            a_A   = np.random.normal(self.para_A[1], 0)
            b_A   = np.random.normal(self.para_A[2], 0)
            self.s_0_A = np.random.normal(2, 0)
            self.IDM_A = IDM(self.v_0, T_A, a_A, b_A, self.s_0_A)

    def observe(self):         
        output = []
        output.append(self.E[self.t,0]-self.B[self.t,0])           #E position - B position: longitudinal
        output.append(self.A[self.t,0]-self.B[self.t,0])           #A position - B position: longitudinal
        output.append(self.F[self.t,0]-self.B[self.t,0])           #F position - B position: longitudinal
        output.append(self.E[self.t,1])                              #E speed: longitudinal
        output.append(self.A[self.t,1])                              #A speed: longitudinal
        output.append(self.F[self.t,1])                              #F speed: longitudinal
        output.append(self.B[self.t,2])                              #B acc
        output.append(self.B[self.t,5])                              #B yaw rate
        output.append(self.B[self.t,1])                              #B speed
        output.append(self.B[self.t,4])                              #B yaw angle
        output.append(self.B[self.t,3] - self.lane_wid*0.5)          #B lateral position lane - target
        
        # 1x63
        output = np.asarray(output).reshape(1,-1)
        '''
        temp = np.isinf(output)
        if len(temp[temp == True])!=0:
            raise print('Inf warning')

        temp = np.isnan(output)
        if len(temp[temp == True])!=0:
            raise print('NaN warning')
        '''
        return output, self.t

    def reward(self):
        # Lane change
        sigma_LC = 1
        r1 = np.exp(-(self.B[self.t+1,3] - self.lane_wid*0.5)**2/(2*sigma_LC**2))    # small differences in large errors, but large improvement in small errors
        
        return r1
    
    def read(self): 
        return self.Dat
    
    def model(self):
        return self.EIDM_B

    def run(self, act):      # act[acc, yaw rate]

        if self.loop_end == False:
            ## vehicle B
            self.B[self.t+1,2] = act[0]                                                          #acc
            self.B[self.t+1,1] = self.B[self.t,1] + 0.1 * self.B[self.t+1,2]                     #speed
            self.B[self.t+1,5] = act[1]                                                          #yaw rate
            self.B[self.t+1,4] = self.B[self.t,4] + 0.1 * self.B[self.t+1,5]/10                  #yaw angle     

            self.B[self.t+1,0] = self.B[self.t,0] + 0.1 * self.B[self.t+1,1] * np.cos(self.B[self.t+1,4])  #longitudinal position                                    #speed
            self.B[self.t+1,3] = self.B[self.t,3] + 0.1 * self.B[self.t+1,1] * np.sin(self.B[self.t+1,4])  #lateral position
                
            ## vehicle D
            if self.B[self.t+1,3]>self.lane_wid: 
                v_tem      = self.D[self.t,1]
                v_delt_tem = self.D[self.t,1] - self.B[self.t,1]
                s_tem      = self.B[self.t,0] - self.D[self.t,0] - self.veh_len
                self.D[self.t+1,2] = self.IDM_D(v_tem, v_delt_tem, s_tem)                        #acc
                self.D[self.t+1,1] = self.D[self.t,1] + 0.1 * self.D[self.t+1,2]                 #speed
                self.D[self.t+1,0] = self.D[self.t,0] + 0.1 * self.D[self.t+1,1]                 #position
            else:
                v_tem      = self.D[self.t,1]
                v_delt_tem = self.D[self.t,1] - self.F[self.t,1]
                s_tem      = self.F[self.t,0] - self.D[self.t,0] - self.veh_len
                self.D[self.t+1,2] = self.IDM_D(v_tem, v_delt_tem, s_tem)                        #acc
                self.D[self.t+1,1] = self.D[self.t,1] + 0.1 * self.D[self.t+1,2]                 #speed
                self.D[self.t+1,0] = self.D[self.t,0] + 0.1 * self.D[self.t+1,1]                 #position
 
            ## Vehicle A
            v_tem      = self.A[self.t,1]
            v_delt_tem = self.A[self.t,1] - self.E[self.t,1]
            s_tem      = self.E[self.t,0] - self.A[self.t,0] - self.veh_len
            a_E = self.IDM_A(v_tem, v_delt_tem, s_tem)                                                         #acc

            v_tem      = self.A[self.t,1]
            v_delt_tem = self.A[self.t,1] - self.B[self.t,1]
            s_tem      = self.B[self.t,0] - self.A[self.t,0] - self.veh_len
            a_B = self.IDM_A(v_tem, v_delt_tem, s_tem)                                                         #acc

            alpha = (self.B[self.t+1,3] - (self.lane_wid))/1
            if alpha > 1:
                alpha = 1
            elif alpha < 0:
                alpha = 0
            self.A[self.t+1,2] = a_E*alpha + a_B*(1-alpha)
            self.A[self.t+1,1] = self.A[self.t,1] + 0.1 * self.A[self.t+1,2]                 #speed
            self.A[self.t+1,0] = self.A[self.t,0] + 0.1 * self.A[self.t+1,1]                 #position
            
            ## Vehicle C
            v_tem      = self.C[self.t,1]
            v_delt_tem = self.C[self.t,1] - self.A[self.t,1]
            s_tem      = self.A[self.t,0] - self.C[self.t,0] - self.veh_len
            self.C[self.t+1,2] = self.IDM_C(v_tem, v_delt_tem, s_tem)                        #acc
            self.C[self.t+1,1] = self.C[self.t,1] + 0.1 * self.C[self.t+1,2]                 #speed
            self.C[self.t+1,0] = self.C[self.t,0] + 0.1 * self.C[self.t+1,1]                 #position
            
            
            ## Vehicle G, H
            v_tem      = self.G[self.t,1]
            v_delt_tem = self.G[self.t,1] - self.C[self.t,1]
            s_tem      = self.C[self.t,0] - self.G[self.t,0] - self.veh_len
            self.G[self.t+1,2] = self.IDM_G(v_tem, v_delt_tem, s_tem)                        #acc
            self.G[self.t+1,1] = self.G[self.t,1] + 0.1 * self.G[self.t+1,2]                 #speed
            self.G[self.t+1,0] = self.G[self.t,0] + 0.1 * self.G[self.t+1,1]                 #position

            v_tem      = self.H[self.t,1]
            v_delt_tem = self.H[self.t,1] - self.G[self.t,1]
            s_tem      = self.G[self.t,0] - self.H[self.t,0] - self.veh_len
            self.H[self.t+1,2] = self.IDM_H(v_tem, v_delt_tem, s_tem)                        #acc
            self.H[self.t+1,1] = self.H[self.t,1] + 0.1 * self.H[self.t+1,2]                 #speed
            self.H[self.t+1,0] = self.H[self.t,0] + 0.1 * self.H[self.t+1,1]                 #position
           
            
            #lane change
            # before crossing lane-marking
            if self.B[self.t+1,3] > self.lane_wid:
                if self.Dat[self.t,12] < (self.Dat[self.t,3] + self.veh_len + self.s_0_A):       #if current B position < A position + vehicle_length
                    LC_temp = 0
                elif self.Dat[self.t,0] < (self.Dat[self.t,12] + self.veh_len + self.s_0_B):     #if current E position < B position + vehicle_length
                    LC_temp = 0
                else:
                    a_B_old_temp = self.B[self.t+1,2]
                    a_A_old_temp = self.A[self.t+1,2]
                    a_D_old_temp = self.D[self.t+1,2]

                    #B new
                    v_tem        = self.Dat[self.t,13]                                       # B current speed
                    s_tem        = self.Dat[self.t,0] - self.Dat[self.t,12] - self.veh_len   # E current position - B current position - vehicle_length
                    cv_arg       = (self.Dat[self.t,1] - self.Dat[self.t,13], self.Dat[self.t,2] - self.Dat[self.t,14]) # E speed - B speed, E acc - B acc
                    a_B_new_temp = self.EIDM_B(s_tem, v_tem, self.v_0, cv_arg)
                    

                    #A new
                    v_tem        = self.Dat[self.t,4]                                       # A current speed
                    v_delt_tem   = self.Dat[self.t,4] - self.Dat[self.t,13]                 # A current speed - B current speed
                    s_tem        = self.Dat[self.t,12] - self.Dat[self.t,3] - self.veh_len  # B current position - A current position - vehicle_length
                    a_A_new_temp = self.IDM_A(v_tem, v_delt_tem, s_tem) 

                    #D new
                    v_tem        = self.Dat[self.t,16]                                          # D current speed
                    v_delt_tem   = self.Dat[self.t,16] - self.Dat[self.t,10]                    # D current speed - F current speed
                    s_tem        = self.Dat[self.t,9] - self.Dat[self.t,15] - self.veh_len      # F current position - D current position - vehicle_length
                    a_D_new_temp = self.IDM_D(v_tem, v_delt_tem, s_tem) 

                    #MOBIL
                    LC_temp  = self.MOBIL_B(a_B_new_temp, a_B_old_temp, a_A_new_temp, a_A_old_temp, a_D_new_temp, a_D_old_temp)
            else:
                LC_temp = 0
                
            #self.Dat[self.t+1,0:3]   = self.E[self.t+1,:]                          # position, speed, acc
            self.Dat[self.t+1,3:6]   = self.A[self.t+1,:] 
            self.Dat[self.t+1,6:9]   = self.C[self.t+1,:] 
            #self.Dat[self.t+1,9:12]  = self.F[self.t+1,:] 
            self.Dat[self.t+1,12:15] = self.B[self.t+1,0:3] 
            self.Dat[self.t+1,15:18] = self.D[self.t+1,:] 
            self.Dat[self.t+1,18:21] = self.G[self.t+1,:] 
            self.Dat[self.t+1,21:24] = self.H[self.t+1,:] 
            self.Dat[self.t+1,24]    = LC_temp
            self.Dat[self.t+1,25:28] = self.B[self.t+1,3:6]    
            
            self.lane_change    = LC_temp

            reward  = self.reward()
            
            if self.t + 1 == self.Time_len - 1:   # current time is 319
                self.loop_end = True
            self.t += 1
            
        else:
            print('Warning: the simulation loop is finished, please reset the environment!')

        return reward, self.loop_end
