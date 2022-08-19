import numpy as np
from pyfilter.lakman import ExtendedKalmanFilter

def HJacobian(x):
    H = [[1, 0, 0 ,0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0]]
    return np.array(H)

def Hx(x, z):
    # we measure exactly the state values
    return z

def substract_angles(target, source):
    return np.arctan2(np.sin(target-source), np.cos(target-source))

def residual(a, b):
    y = a - b
    y[2] = substract_angles(a[2], b[2])
    return y

class AntEKF(ExtendedKalmanFilter):
    # dim_x: x, y, a, v, w    
    X_SIZE = 5
    # dim_z: x, y, a
    Z_SIZE = 3
    
    '''
    start x - [x, y, a, v, w]
    p - probabil of detection
    R_diag - [rx, ry, ra]
    Q_diag - [qx, qy, qa, qv, qw]
    dt - 1/fps
    '''
    def __init__(start_x, p, Rdiag, Q_diag, dt):
        super(ExtendedKalmanFilter, self).__init__(X_SIZE, Z_SIZE)
        
        self.dt = dt 
        self.x = start_x
        self.P = np.eye(X_SIZE) * (1 - p) # maybe not good idea
        self.R = np.diag(R_diag)
        self.Q = np.diag(Q_diag)
        
        self.no_update_steps = 0
                
        
    def predict(self, u = 0):
        # just write our movement equations
        
        self.x[0] = self.x[0] + self.x[3] * np.cos(self.x[2]) * self.dt
        self.x[1] = self.x[1] + self.x[3] * np.sin(self.x[2]) * self.dt
        self.x[2] = self.x[2] + self.x[4] * self.dt
        self.x[3] = self.x[3] # no change
        self.x[4] = self.x[4] # no change
        
        self.F = np.array([[1, 0, -np.sin(self.x[2]) * self.x[3] * self.dt, np.cos(self.x[2]) * self.dt, 0],
                      [0, 1, np.cos(self.x[2]) * self.x[3] * self.dt, np.sin(self.x[2]) * self.dt, 0],
                      [0, 0, 1, 0, self.dt],
                      [0, 0, 0, 1, 0],
                      [0 0, 0, 0, 1]])
        
        self.P = F @ self.P @ F.T + self.Q
                        
        self.no_update_steps+=1
        # TODO: add x to history
        
    '''
    new_value - [x, y, a]
    '''
    def update2(self, new_value):
        z = np.array([new_value])
        self.update(z, HJacobian = HJacobian, Hx = Hx, residual = residual, hx_args=(z))
        
        self.no_update_steps = 0
        # TODO: rewrite last history element

        
'''
Calculates pairvise mahalanobis distances between new values x, old values y with correspondence to old values covariations
    x - new values, array of [K, N]
    y - old values, array of [M, N]
    Sm - inverted cov matrixes, array of [M, N, N]
returns
    array [K, M]
'''
def multi_mahalanobis(x, y, Sm):
    
    xx = np.tile(x, (y.shape[0],1))
    yy = np.repeat(y, x.shape[0], axis = 0)
                
    SSm = np.repeat(Sm, x.shape[0], axis = 0)        
    
    d = xx - yy        
    de = np.expand_dims(d, 1)
    dee = np.expand_dims(d, 2)        
    
    D = np.matmul(de, SSm)            
    D = np.sqrt( np.matmul( D, dee))    
    D = D.reshape((x.shape[0], y.shape[0]))
    
    return D

class multiEKF(object):
    
    
    def __init__(self, start_values, R_diag, Q_diag, dt, mahalanobis_thres):
        
        self.mahalanobis_thres = mahalanobis_thres
        
        self.R_diag = R_diag
        self.Q_diag = Qdiag
        self.dt = dt
        
        EKFS = []
        for i in start_values.shape[0]:
            
            ekf = AntEKF(start_values[i][1:], start_values[i][0], self.R_diag, self.Q_diag, self.dt)                                    
            EKFS.append(ekf)                        

    '''
    new values - [[p, x, y, a]]
    '''
    def proceed(self, new_values):
        # 1. predict previous
        for ekf in self.EKFS:
            ekf.predict()
        
        # 2. calc correspondence
        old_values = self.get_all_ants_data_as_array()
        
        inv_covs = np.array([np.linalg.inv(ekf.P[:3,:3]) for ekf in self.EKFS])
        correspondence_matrix = multi_mahalanobis(new_values[:,1:], old_values, inv_covs)
        
        # store indexes of all ants, and then delete those which is taken for update, the rest will be new ants
        new_objects = list(range(new_values.shape[0]))
        
        # 3. update where correspondence is
        while True:
            # find minimal value from matrix
            minimal_distance = np.unravel_index(np.argmin(correspondence_matrix, axis=None), correspondence_matrix.shape)
            
            if correspondence_matrix[minimal_distance] > self.mahalanobis_thres:
                break # no more ants satisfy threshold
            
            ekf_ind = minimal_distance[1]
            val_ind = minimal_distance[0]
            
            # update filter
            self.EKFS[ekf_ind].update2(new_values[val_ind, 1:])
            
            # 'remove' values from matrix
            correspondence_matrix[val_ind,:] = np.inf
            correspondence_matrix[:,ekf_ind] = np.inf
            # and from 'new_objects'
            new_objects.remove(val_ind)
        
        # 4. add new filters for new objects
        for ind in new_objects:            
            new_x = np.zeros(5)
            new_x[:3] = new_values[ind][1:]
            ekf = AntEKF(new_x, new_values[ind][0], self.R_diag, self.Q_diag, self.dt)                                    
            EKFS.append(ekf)
                
        # 5. forget bad filters (long no update, huge covs, etc.) 
        
    def get_all_ants_data_as_array(self):
        ants = []
        for ekf in self.EKFS:
            ants.append(ekf.x)
        return np.array(ants)
        
