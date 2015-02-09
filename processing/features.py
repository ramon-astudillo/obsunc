#!/usr/bin/python

import numpy as np
import pdb
import sys # for stdout.write 
import time # for clock
import signal as sip

#############################################
# MEL-FREQUENCY CEPSTRAL COEFFICIENTS
#############################################
    
class mfcc():

    '''
    This is run outside the file loop to initialize features
    '''

    def __init__(self, sample_freq, nfft, numchans, numceps, ceplifter, 
                 usepow=1):

        #
        self.usepow=usepow

        #
        # MEL-FILTERBANK WEIGTHS
        #
    
        # EQUALY SPACED BINS IN MEL-DOMAIN (freq bins under fs/2) 
        cent_mf = np.linspace(0,1127*np.log(1 + 0.5*sample_freq/700),numchans+2)
        # Transform them back to Fourier domain
        cent_f  = (np.exp(cent_mf/1127)-1)*700
        
        # CREATE WEIGHT MATRIX OF FILTERBANK
        self.W = np.zeros([numchans, nfft/2+1])
        # For each filter center 
        for m in range(1,len(cent_f)-1):
            # For each frequency bin
            for k in range(0,nfft/2+1):
                # FIND WEIGHT
                # Frecuency corresponding to bin k
                frec = k*sample_freq/nfft
                # If it belongs to ascending ramp of the filter
                if (frec <= cent_f[m]) and (frec >= cent_f[m-1]):
                    self.W[m-1,k] = (frec-cent_f[m-1])/(cent_f[m]-cent_f[m-1])
                # if it belongs to descending ramp of the filter
                elif (frec <= cent_f[m+1]) and (frec >= cent_f[m]):
                    self.W[m-1,k] = 1-((frec-cent_f[m])/(cent_f[m+1]-cent_f[m]))
                else:
                    # Does not belong to filter    
                    self.W[m-1,k] = 0
        #    
        # DCT WEIGTHS
        #
    
        # NORMAL DCT FROM SIGNAL
        # NOTE: This is not exactly the same using Matlabs DCT()
        self.T = np.sqrt(2.0/numchans) * np.cos( np.pi/numchans * np.outer(range(0,numceps+1),(np.array(range(1,numchans+1))- 0.5)))
        
        # CEPLIFTER
        if ceplifter > 0:
            lift_matr = 1 + ceplifter/2 * np.sin(np.transpose(range(0,numceps+1))*np.pi/ceplifter)
            # Multiply by ceplifter matrix
            self.T = np.dot(np.diag(lift_matr),self.T)
    

    def extract(self, X):
        '''
        Feature extraction
        '''
        # Amplitude or Power based MFCC
        if self.usepow:
            M = np.dot(self.W, np.absolute(X)**2)
        else:
            M = np.dot(self.W, np.absolute(X))
        # Floor Mel channels
        M[M<1e-6] = 1e-6
        return np.dot(self.T, np.log(M))

    def extract_up(self, mu_X, Lambda_X, diagcov_flag=1):
        '''
        Feature extraction for c. s. complex Gaussian uncertain STFT
        '''
        if self.usepow:    
            # PERIODOGRAM ESTIMATION OF PSD
            mu_P     = np.absolute(mu_X)**2 + Lambda_X
            dSigma_P = Lambda_X *(2*np.absolute(mu_X)**2 + Lambda_X)
        else:    
            if np.all(Lambda_X == 0):
                mu_P     = np.absolute(mu_X)
                dSigma_P = np.zeros(mu_P.shape) 
            else:
                mu_P     = np.absolute(sip.MMSE_STSA(mu_X,Lambda_X))
                dSigma_P = np.absolute(mu_X)**2 + Lambda_X - mu_P**2

        # If covariance after Mel-filterbank ingnored
        if diagcov_flag:
    
            # MEL-FILTERBANK
            mu_M     = np.dot(self.W,mu_P)
            dSigma_M = np.dot(self.W**2,dSigma_P)
            
            # LOGARITHM
            dSigma_L = np.log(np.divide(dSigma_M,mu_M**2) + 1)
            mu_L     = np.log(mu_M) - 0.5 * dSigma_L
            
            # DISCRETE COSNINE TRANSFORM
            mu_C     = np.dot(self.T,mu_L)
            Sigma_C  = np.dot(self.T**2,dSigma_L)
    
        else:    
    
            # INITIALIZATION
            L            = mu_X.shape[1]
            [nceps,nmel] = self.T.shape           
    
            # MEL-FILTERBANK
            mu_M         = np.dot(self.W,mu_P)
            mu_C         = np.zeros([nceps, L])
    #        Sigma_C      = np.zeros([nceps, nceps, L])    # Fullcov cepstra version    
            Sigma_C      = np.zeros([nceps, L])        
            for l in np.arange(0,L):
                Sigma_M = np.dot(self.W,
                                 np.dot(np.diagflat(dSigma_P[:,l:l+1]),
                                        np.transpose(self.W)))
                # LOGARITHM
                Sigma_L = np.log(np.divide(Sigma_M, np.dot(mu_M[:,l:l+1],
                                                           mu_M[:,l:l+1].T))
                                           + 1) 
                mu_L    = (np.log(mu_M[:,l:l+1]) 
                           -0.5*np.diag(Sigma_L)[:,None])
                # DCT
                Sigma_C[:,l:l+1]   = np.diag(np.dot(self.T,
                                     np.dot(Sigma_L,self.T.T)))[:,None]
                mu_C[:,l:l+1]      = np.dot(self.T,mu_L)
   
        return [mu_C,Sigma_C]


    # Uncertain MFCCs 
    def extract_up_mc(self, mu_X, Lambda_X, max_samples=1e3, max_simult_samples=1):
        '''
        Feature extraction for c. s. complex Gaussian uncertain STFT, Monte Carlo solution
        '''
        # Replicate signal to trade main loop iterations for memory footprint
        mu_X     = np.tile(mu_X, (1, max_simult_samples))
        Lambda_X = np.tile(Lambda_X, (1, max_simult_samples))
    
        # INITIALIZATION
        [K,L]        = mu_X.shape
        [nceps,nmel] = self.T.shape           
        mu_C         = np.zeros([nceps, L])
        mu_C2        = np.zeros([nceps, L]) # Second order moment
    
        print "\nComputing Monte Carlo simulation"
        
        # For each sample            
        avtime   = 0
        for i in np.arange(0,max_samples):
    
            # INFO
            init_time = time.clock() 
    
            # Draw circularly symmetric complex Gaussian sample
            sample_X = np.real(mu_X) + np.sqrt(Lambda_X/2)*np.random.randn(K,L) + 1j*(np.imag(mu_X) + np.sqrt(Lambda_X/2)*np.random.randn(K,L))
    
            # Transform sample
            sample_x = self.extract(sample_X)
    
            # Update statistics
            mu_C     = (mu_C*i + sample_x)/(i+1) 
            mu_C2    = (mu_C2*i + sample_x**2)/(i+1)
    
            # INFO
            avtime   = (avtime*i + (time.clock() - init_time))/(i+1)
    
            # Update a progress bar of 50 slots
            if not i % np.floor(max_samples/50):
                progress = np.floor(50*(i*1.0)/max_samples)    
                sys.stdout.write('\r%d%% [%-50s] %2.2f samples/sec'%(progress*(100/50),'='*progress + '>',max_simult_samples/avtime))
                sys.stdout.flush()
    
        sys.stdout.write('\r%d%% [%-50s]\n\n'%(50*(100/50),'='*50))
    
        # Merge the stats for the max_simult_samples copies
        total_mu_C  = np.zeros([nceps,L/max_simult_samples])
        total_mu_C2 = np.zeros([nceps,L/max_simult_samples])
        # Loop    
        for s in np.arange(0,max_simult_samples):
            # Range of coresponding samples
            idx         = s*(L/max_simult_samples) + np.arange(0,(L/max_simult_samples))
            # Add to total    
            total_mu_C  = total_mu_C  + mu_C[:,idx]/max_simult_samples
            total_mu_C2 = total_mu_C2 + mu_C2[:,idx]/max_simult_samples
        
        #
        mu_C = total_mu_C
        # Compute variance
        Sigma_C = total_mu_C2 - total_mu_C**2    
        
        return [mu_C,Sigma_C]

    def cms(self,x):
        return x - np.mean(x,1)[:,None]

    def cms_up(self,mu_x,Sigma_x):
    
        L        = mu_x.shape[1]    
        mu_CM    = np.mean(mu_x,1)
        Sigma_CM = np.mean(Sigma_x,1)/L
    
        mu_y     = mu_x    - mu_CM[:,None]
        Sigma_y  = Sigma_x + Sigma_CM[:,None] - 2*Sigma_x/L
    
        return [mu_y,Sigma_y]


#################################################
# DELTAS AND ACCELERATIONS
#################################################

def deltas(x, window=2, weighted=1):
    '''
    Deltas and Accelerations
    '''

    [I,L] = x.shape
    denom = 2*np.sum(np.arange(1,window+1)**2)
    y     = np.zeros([I,L])
    
    # Create indices statically, out of bounds are floored or ceiled
    idx     = np.arange(-window,window+1)[:,None] + np.arange(L)[None,:]
    idx[idx < 0]   = 0    
    idx[idx > L-1] = L-1 
 
    if weighted: 
        weigths    = np.arange(-window,window+1)[:,None] 
    else:
        weigths    = np.concatenate((-np.ones(window),np.array([0]),
                                     np.ones(window)))[:,None]

    for i in np.arange(0,I):
        y[i,:] = np.sum(weigths*x[i,idx],0)/denom 
    return y 

def deltas_up(mu_x, Sigma_x, window=2, weigthed=1):
    '''
    Deltas and Accelerations
    '''

    # Initialization
    [I,L]    = mu_x.shape
    denom    = 2*np.sum(np.arange(1,window+1)**2)
    mu_y     = np.zeros([I,L])
    Sigma_y  = np.zeros([I,L])

    # Create indices statically, out of bounds are floored or ceiled
    idx            = (np.arange(-window,window+1)[:,None] 
                      + np.arange(L)[None,:])
    idx[idx < 0]   = 0    
    idx[idx > L-1] = L-1 
    # Compute weighths
    if weigthed: 
        weigths    = np.arange(-window,window+1)[:,None] 
    else:
        weigths    = np.concatenate((-np.ones(window),np.array([0]),
                                     np.ones(window)))[:,None]
    # Compute deltas
    for i in np.arange(0,I):
        mu_y[i,:]    = np.sum(weigths*mu_x[i,idx],0)/denom 
        Sigma_y[i,:] = np.sum((weigths**2)*Sigma_x[i,idx],0)/(denom**2)

    return [mu_y,Sigma_y]
