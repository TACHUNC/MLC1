import math
import numpy as np
from copy import deepcopy
from typing import Callable
# from sklearn.gaussian_process.kernels import RBF,Matern
# from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import norm, qmc

class GP_model:   
    def __init__(self, X, Y, kernel, multi_hyper, var_out=True):
        # GP variable definitions
        self.X, self.Y, self.kernel = X, Y, kernel
        self.n_point, self.nx_dim   = X.shape[0], X.shape[1]
        self.ny_dim                 = 1
        self.multi_hyper            = multi_hyper
        self.var_out                = var_out
        
        # normalize data
        self.X_mean, self.X_std     = np.mean(X, axis=0), np.std(X, axis=0)
        self.Y_mean, self.Y_std     = np.mean(Y, axis=0), np.std(Y, axis=0)
        self.X_norm, self.Y_norm    = (X-self.X_mean)/self.X_std, (Y-self.Y_mean)/self.Y_std
        
        # determine hyperparameters
        self.hypopt, self.invKopt   = self.determine_hyperparameters()        
    
    def Cov_mat(self, kernel, X_norm, W, sf2):
    
        if kernel == 'RBF':
            dist       = cdist(X_norm, X_norm, 'seuclidean', V=W)**2 
            cov_matrix = sf2*np.exp(-0.5*dist)
            return cov_matrix
        elif kernel == 'MATERN_52':
            # Matern 5/2 Kernel
            dist = cdist(X_norm, X_norm, 'seuclidean', V=W)
            sqrt_5_dist = np.sqrt(5) * dist
            cov_matrix = sf2 * (1 + sqrt_5_dist + (5/3) * dist**2) * np.exp(-sqrt_5_dist)
            return cov_matrix
    
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")

    def calc_cov_sample(self,xnorm,Xnorm,ell,sf2):   
        # internal parameters
        nx_dim = self.nx_dim

        dist = cdist(Xnorm, xnorm.reshape(1,nx_dim), 'seuclidean', V=ell)**2
        cov_matrix = sf2 * np.exp(-.5*dist)

        return cov_matrix               
    
    def negative_loglikelihood(self, hyper, X, Y):
        # internal parameters
        n_point, nx_dim = self.n_point, self.nx_dim
        kernel          = self.kernel
        
        W               = np.exp(2*hyper[:nx_dim])   # W <=> 1/lambda
        sf2             = np.exp(2*hyper[nx_dim])    # variance of the signal 
        sn2             = np.exp(2*hyper[nx_dim+1])  # variance of noise

        K       = self.Cov_mat(kernel, X, W, sf2)  # (nxn) covariance matrix (noise free)
        K       = K + (sn2 + 1e-8)*np.eye(n_point) # (nxn) covariance matrix
        K       = (K + K.T)*0.5                    # ensure K is simetric
        L       = np.linalg.cholesky(K)            # do a cholesky decomposition
        logdetK = 2 * np.sum(np.log(np.diag(L)))   # calculate the log of the determinant of K the 2* is due to the fact that L^2 = K
        invLY   = np.linalg.solve(L,Y)             # obtain L^{-1}*Y
        alpha   = np.linalg.solve(L.T,invLY)       # obtain (L.T L)^{-1}*Y = K^{-1}*Y
        NLL     = np.dot(Y.T,alpha) + logdetK      # construct the NLL

        return NLL
    
    def determine_hyperparameters(self):  
        # internal parameters
        X_norm, Y_norm  = self.X_norm, self.Y_norm
        nx_dim, n_point = self.nx_dim, self.n_point
        kernel, ny_dim  = self.kernel, self.ny_dim
        Cov_mat         = self.Cov_mat
        
        lb               = np.array([-4.]*(nx_dim+1) + [-8.])  # lb on parameters (this is inside the exponential)
        ub               = np.array([4.]*(nx_dim+1) + [ -2.])   # lb on parameters (this is inside the exponential)
        bounds           = np.hstack((lb.reshape(nx_dim+2,1),
                                      ub.reshape(nx_dim+2,1)))
        
        multi_start      = self.multi_hyper                   # multistart on hyperparameter optimization
        
        sampler = qmc.Sobol(d=nx_dim + 2,scramble=False)
        multi_startvec = sampler.random(multi_start)
        
        # multi_startvec   = sobol_seq.i4_sobol_generate(nx_dim + 2,multi_start)

        options  = {'disp':False,'maxiter':10000}          # solver options
        hypopt   = np.zeros((nx_dim+2, ny_dim))            # hyperparams w's + sf2+ sn2 (one for each GP i.e. output var)
        localsol = [0.]*multi_start                        # values for multistart
        localval = np.zeros((multi_start))                 # variables for multistart

        invKopt = []
        # --- loop over outputs (GPs) --- #
        for i in range(ny_dim):    
            # --- multistart loop --- # 
            for j in range(multi_start):
                hyp_init    = lb + (ub-lb)*multi_startvec[j,:]
                # --- hyper-parameter optimization --- #
                res = minimize(self.negative_loglikelihood,hyp_init,args=(X_norm,Y_norm[:])\
                               ,method='SLSQP',options=options,bounds=bounds,tol=1e-12)
                localsol[j] = res.x
                localval[j] = res.fun

            # --- choosing best solution --- #
            minindex    = np.argmin(localval)
            hypopt[:,i] = localsol[minindex]
            ellopt      = np.exp(2.*hypopt[:nx_dim,i])
            sf2opt      = np.exp(2.*hypopt[nx_dim,i])
            sn2opt      = np.exp(2.*hypopt[nx_dim+1,i]) + 1e-8

            # --- constructing optimal K --- #
            Kopt        = Cov_mat(kernel, X_norm, ellopt, sf2opt) + sn2opt*np.eye(n_point)
            # --- inverting K --- #
            invKopt     += [np.linalg.solve(Kopt,np.eye(n_point))]

        return hypopt, invKopt
    
    def predict(self, x):
        nx_dim                   = self.nx_dim
        kernel, ny_dim           = self.kernel, self.ny_dim
        hypopt, Cov_mat          = self.hypopt, self.Cov_mat
        stdX, stdY, meanX, meanY = self.X_std, self.Y_std, self.X_mean, self.Y_mean
        calc_cov_sample          = self.calc_cov_sample
        invKsample               = self.invKopt
        Xsample, Ysample         = self.X_norm, self.Y_norm
        var_out                  = self.var_out

        xnorm = (x - meanX)/stdX
        mean  = np.zeros(ny_dim)
        var   = np.zeros(ny_dim)

        for i in range(ny_dim):
            invK           = invKsample[i]
            hyper          = hypopt[:,i]
            ellopt, sf2opt = np.exp(2*hyper[:nx_dim]), np.exp(2*hyper[nx_dim])

            k       = calc_cov_sample(xnorm,Xsample,ellopt,sf2opt)
            mean[i] = np.matmul(np.matmul(k.T,invK),Ysample[:])
            var[i]  = max(0, sf2opt - np.matmul(np.matmul(k.T,invK),k)) 

        # --- compute un-normalized mean --- #    
        mean_sample = mean*stdY + meanY
        var_sample  = var*stdY**2
        
        if var_out:
            return mean_sample, np.sqrt(var_sample)
        else:
            return mean_sample.flatten()[0]
    
    def return_hyperparams(self):
        return np.exp(2*self.hypopt[:self.nx_dim])
        
def CDF(Y_best,mean,std_dev):
    z = (Y_best - mean)/std_dev
    return norm.cdf(z)

def PDF(Y_best,mean,std_dev):
    z = (Y_best - mean)/std_dev
    return norm.pdf(z)

def to_unit_cube(x, lb, ub):
    return (x-lb) / (ub-lb)

def from_unit_cube(x,lb,ub):
    return x*(ub-lb) + lb

def latin_hypercube(n_pts, dim):
    x = np.zeros((n_pts, dim))
    centers = (1.0+2.0*np.arange(0.0,n_pts)) / (2*n_pts) 
    for i in range(dim):
        x[:, i] = centers[np.random.permutation(n_pts)]
        
    pert = np.random.uniform(-1,1,(n_pts,dim))
    pert = pert/(2*n_pts)
    x += pert
    return x

class BO_EI:
    def __init__(self, 
                 fun_list:list[Callable], 
                 x_init:np.array, 
                 bounds:np.array,
                 budget:float,
                 n_init:int=4) -> None:
        
        # transform shape from (n,) -> (1,n)
        if x_init.ndim == 1:
            x_init = x_init.reshape(1,-1)
        
        self.n_eq       = len(fun_list)
        self.n_dim      = bounds.shape[0]
        self.bounds     = bounds
        self.lb,self.ub = bounds[:,0], bounds[:,1]
        self.fun_list   = fun_list
        self.budget     = budget
        
        # if only one initial point given, sampling more 
        # for GP training
        if x_init.shape[0] == 1:
            x_sample = latin_hypercube(n_init, self.n_dim)
            x_sample = from_unit_cube(x_sample, self.lb, self.ub)
            x_init = np.vstack((x_init,x_sample))
        
        Y = np.zeros((x_init.shape[0],self.n_eq))
        
        for i_fun, fun in enumerate(fun_list):
            for i_data, x in enumerate(x_init):
                obj_val = fun(x.flatten())
                Y[i_data,i_fun] = obj_val
                
        self.X  = x_init
        self.Y  = Y
        
        # tolerances and counters
        _n_cand = 500*self.n_dim
        _pow = math.ceil(math.log2(_n_cand))
        self.n_cand = min(2**_pow, 4096)
        self.failtol = math.ceil(max(4.0, self.n_dim))
        self.succtol = 3
        self.n_evals = x_init.shape[0]
    
        # truest region sizes
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.length_init = 0.8
        
        self.train_surrogate()
    
    def train_surrogate(self):
        surrogate_list = []
        for i in range(self.n_eq):
            GP = GP_model(self.X,self.Y[:,i],'MATERN_52',8,)
            surrogate_list.append(GP)
            
        self.surrogate_list = surrogate_list
    
    def restart(self):
        self.failcount = 0
        self.succount = 0
        self.length = self.length_init
    
    def adjust_length(self, fX_next:np.array):
        if (fX_next[0,0] < np.min(self.Y[:,0])) and (fX_next[0,1:]<=0).all():
            self.succount += 1
            self.failcount = 0
        else:
            self.succount = 0
            self.failcount += 1
            
        if self.succount == self.succtol:
            self.length = min(2.0*self.length, self.length_max)
            self.succount = 0
        elif self.failcount == self.failtol:
            self.length /= 2.0
            self.failcount = 0
        
    def create_candidates(self,X, Y,length):
        self.train_surrogate()
        GP_obj = self.surrogate_list[0]
        
        # shape (1, n_dim)
        length_scale_obj = GP_obj.return_hyperparams().reshape(1,-1)
        length_scale_obj /= length_scale_obj.mean() 
        length_scale_obj /= np.prod(np.pow(length_scale_obj,1/self.n_dim))
        
        _constraints = Y[:,1:]
        _feasible_ind = (_constraints <= 0).all(axis=1)
        # if any of the solution is feasible
        if _feasible_ind.any(): 
            _best_ind = np.argmin(Y[_feasible_ind,0])
            x_center = X[_best_ind,:] 
        # else none of them are feasible, make the center to be the point having least violation.
        else:
            Y[-_feasible_ind,0]
            _best_ind = np.argmin(Y[:,1:].sum(axis=1))
            x_center = X[_best_ind,:]
            
        lb = np.clip(x_center - length_scale_obj * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + length_scale_obj * length / 2.0, 0.0, 1.0)
        
        # perturbation mask
        sampler = qmc.Sobol(d=self.n_dim,scramble=False)
        pert = sampler.random(self.n_cand)
        pert = lb + (ub-lb)*pert
        pert_prob = min(20/self.n_dim, 1.0)
        mask = np.random.rand(self.n_cand, self.n_dim) <= pert_prob
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.n_dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center * np.ones((self.n_cand, self.n_dim))
        X_cand[mask] = pert[mask]
        
        # Get the corresponding surrogate value
        y_cand, std_cand = np.empty((self.n_cand,1)), np.empty((self.n_cand,1))
        for i_sample, x_sample in enumerate(X_cand):
            y_cand[i_sample,0], std_cand[i_sample,0] = GP_obj.predict(x_sample.flatten())
            
        # expected improvement
        Y_best = Y.min()
        ei = (Y_best - y_cand)*CDF(Y_best,y_cand,std_cand) + std_cand*PDF(Y_best,y_cand,std_cand)

        # Probability of Feasibility
        for _ , GP_cons in enumerate(self.surrogate_list[1:]):
            for i_sample, x_sample in enumerate(X_cand):
                mean ,std = GP_cons.predict(x_sample.flatten())
                Prob_feasible = CDF(0, mean, std)
                ei[i_sample, 0] *= Prob_feasible
            
        return X_cand, ei

    def select_candidates(self, X_cand, y_cand):
        indbest = np.argmax(y_cand)
        X_next = X_cand[indbest, :]
        return X_next
    
    def optimize(self):
        while self.n_evals < self.budget:
            # Initialize parameters
            self.restart()

            # Thompson sample to get next suggestions
            while self.n_evals < self.budget and self.length >= self.length_min:
                # Warp inputs
                X = to_unit_cube(deepcopy(self.X), self.lb, self.ub)
                Y = deepcopy(self.Y)

                X_cand, y_cand = self.create_candidates(X, Y, self.length)
                # best candidate
                X_next = self.select_candidates(X_cand, y_cand)

                # Undo the warping
                X_next = from_unit_cube(X_next, self.lb, self.ub)
                
                # Evaluate best canditate
                Y_next = np.empty((1,self.n_eq))
                for i_fun, fun in enumerate(self.fun_list):
                    Y_next[0, i_fun] = fun(X_next.flatten())
                    
                # Update trust region
                self.adjust_length(Y_next)         

                # Update budget and append data
                self.n_evals += 1
                self.X = np.vstack((self.X, X_next))
                self.Y = np.vstack((self.Y, Y_next))
    
def gp_tr(
        problem,
        bounds,  
        budget, 
        rep
        ): 
    '''
    - problem: 
        data-type: Test_function object
        This is the WO problem
    - bounds: 
        data-type: np.array([[4., 7.], [70., 100.]])
        Flowrate reactand B from 4 to 7 kg/s
        Reactor temperature from 70 to 100 degree Celsius
    - budget: 
        data-type: integer
        budget for WO problem: 20 iterations
    - rep:
        data-type: integer
        repetition to determine the starting point
    '''

    ######### DO NOT CHANGE ANYTHING FROM HERE ############
    x_start = problem.x0[rep].flatten() # numpy nd.array for example [ 6.9 83. ]
    obj = problem.fun_test              # objective function: obj = -(1043.38 * flow_product_R + 20.92 * flow_product_E -79.23 * inlet_flow_reactant_A - 118.34 * inlet_flow_reactant_B) 
    con1 = problem.WO_con1_test         # constraint 1: X_A <= 0.12
    con2 = problem.WO_con2_test         # constraint 2: X_G <= 0.08
    ######### TO HERE ####################################
    
    # list all the functions together
    fun_list = [obj, con1, con2]
    
    algo = BO_EI(fun_list,x_start,bounds,budget)
    algo.optimize()

    ################# ADJUST THESE ######################
    team_names = ['TACHUN, CHEN', 'Member2']
    cids = ['02080022', '01234567']
    return team_names,cids