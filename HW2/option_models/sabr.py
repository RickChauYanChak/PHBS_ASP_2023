# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: Jaehyuk Choi, Enze Zhou
"""

import numpy as np
import pyfeng as pf
import abc

class ModelABC(abc.ABC):
    beta = 1   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr, is_fwd = None, None, None, None

    ### Numerical Parameters
    dt = 0.1
    n_path = 10000

    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0.0, divr=0.0, is_fwd=False):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.beta = beta
        self.intr = intr
        self.divr = divr
        self.is_fwd = is_fwd

    def base_model(self, sigma=None):
        if sigma is None:
            sigma = self.sigma

        if self.beta == 0:
            return pf.Norm(sigma, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
        elif self.beta == 1:
            return pf.Bsm(sigma, intr=self.intr, divr=self.divr, is_fwd=self.is_fwd)
        else:
            raise ValueError(f'0<beta<1 not supported')

    def vol_smile(self, strike, spot, texp=1.0):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        price = self.price(strike, spot, texp, cp=1)
        iv = self.base_model().impvol(price, strike, spot, texp, cp=1)
        return iv

    @abc.abstractmethod
    def price(self, strike, spot, texp=1.0, cp=1):
        """
        Vanilla option price

        Args:
            strike:
            spot:
            texp:
            cp:

        Returns:

        """
        return NotImplementedError

    def sigma_path(self, texp):
        """
        Path of sigma_t over the time discretization

        Args:
            texp:

        Returns:

        """
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        dt = texp / n_dt
        assert texp == tobs[-1]

        Z_t = np.cumsum(np.random.standard_normal((n_dt, self.n_path)) * np.sqrt(dt), axis=0)
        sigma_t = np.exp(self.vov * (Z_t - self.vov/2 * tobs[:, None]))
        sigma_t = np.insert(sigma_t, 0, np.ones(sigma_t.shape[1]), axis=0)

        return sigma_t

    def intvar_normalized(self, sigma_path):
        """
        Normalized integraged variance I_t = \int_0^T sigma_t^2 dt / (sigma_0^2 T)

        Args:
            sigma_path: sigma path

        Returns:

        """

        weight = np.ones(sigma_path.shape[0])
        weight[[0, -1]] = 0.5
        weight /= weight.sum()
        intvar = np.sum(weight[:, None] * sigma_path**2, axis=0)
        return intvar

class ModelBsmMC(ModelABC):
    """
    MC for Bsm SABR (beta = 1)
    """

    beta = 1.0   # fixed (not used)

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t. 

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)

        (2) Simulate S_0, ...., S_T.

        Z = np.random.standard_normal()

        (3) Calculate option prices (vector) for all strikes
        '''

        # Calculate the forward
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac * div_fac

        # Determine the number of time steps
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        dt = texp / n_dt
        assert texp == tobs[-1]

        # Generate the paths of sigma_t
        Z_t1 = np.random.standard_normal((n_dt, self.n_path))
        Z_t = np.cumsum(Z_t1 * np.sqrt(dt), axis=0)
        sigma_t = self.sigma * np.exp(self.vov * (Z_t - self.vov/2 * tobs[:, None]))
        sigma_t = np.insert(sigma_t, 0, np.ones(sigma_t.shape[1]) * self.sigma, axis=0)

        # Generate the paths of W_t for S_t
        Z_t2 = np.random.standard_normal((n_dt, self.n_path))
        W_t = self.rho * Z_t1 + np.sqrt(1 - self.rho**2) * Z_t2

        # Generate the paths of S_t
        sigma_t_disc = sigma_t[:-1, :]
        sqrt_dt = np.sqrt(dt)
        log_St = np.cumsum(sigma_t_disc * sqrt_dt * W_t - sigma_t_disc**2 * dt/2, axis=0)
        S_T = forward * np.exp(log_St[-1, :])

        # Calculate the option prices
        p = disc_fac * np.mean(np.fmax(cp*(S_T - strike[:, None]), 0.0), axis=1)
        return p

class ModelNormMC(ModelBsmMC):
    """
    MC for Normal SABR (beta = 0)
    """

    beta = 0   # fixed (not used)

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t. 

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)

        (2) Simulate S_0, ...., S_T.

        Z = np.random.standard_normal()

        (3) Calculate option prices (vector) for all strikes
        '''

        # Calculate the forward
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac * div_fac
        sigma_0 = self.sigma / spot * forward  # adjust \sigma_0 accordingly

        # Determine the number of time steps
        n_dt = int(np.ceil(texp / self.dt))
        tobs = np.arange(1, n_dt + 1) / n_dt * texp
        dt = texp / n_dt
        assert texp == tobs[-1]

        # Generate the paths of sigma_t
        Z_t1 = np.random.standard_normal((n_dt, self.n_path))
        Z_t = np.cumsum(Z_t1 * np.sqrt(dt), axis=0)
        sigma_t = sigma_0 * np.exp(self.vov * (Z_t - self.vov/2 * tobs[:, None]))
        sigma_t = np.insert(sigma_t, 0, np.ones(sigma_t.shape[1]) * sigma_0, axis=0)

        # Generate the paths of W_t for S_t
        Z_t2 = np.random.standard_normal((n_dt, self.n_path))
        W_t = self.rho * Z_t1 + np.sqrt(1 - self.rho**2) * Z_t2

        # Generate the paths of S_t
        sigma_t_disc = sigma_t[:-1, :]
        sqrt_dt = np.sqrt(dt)
        St = np.cumsum(sigma_t_disc * sqrt_dt * W_t, axis=0)
        S_T = forward + St[-1, :]

        # Calculate the option prices
        p = disc_fac * np.mean(np.fmax(cp*(S_T - strike[:, None]), 0.0), axis=1)
        return p

class ModelBsmCondMC(ModelBsmMC):
    """
    Conditional MC for Bsm SABR (beta = 1)
    """

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t and normalized integrated variance

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) 

        (2) Calculate the equivalent spot and volatility of the BS model

        vol = 
        spot_equiv = 

        (3) Calculate option prices (vector) by averaging the BS prices

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        '''

        # Note that we don't need to calculate the forward in this case because
        # the forward will be treated by the `pf.Bsm()` class

        # Generate the paths of sigma_t and normalized integrated variance
        vol_path = self.sigma * self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) / (self.sigma**2 * texp) * texp
        
        # Calculate the equivalent spot and volatility of the BS model
        vol = self.sigma * np.sqrt((1 - self.rho**2) * I_t)  # Just an example
        spot_equiv = spot * np.exp(self.rho * ((sigma_t - self.sigma) / self.vov - self.rho * self.sigma**2 * texp * I_t / 2))

        # Calculate the option prices
        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        
        return p


class ModelNormCondMC(ModelNormMC):
    """
    Conditional MC for Normal SABR (beta = 0)
    """

    def price(self, strike, spot, texp=1.0, cp=1):
        '''
        Your MC routine goes here.
        (1) Generate the paths of sigma_t and normalized integrated variance

        vol_path = self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) 

        (2) Calculate the equivalent spot and volatility of the Bachelier model

        vol = 
        spot_equiv = 

        (3) Calculate option prices (vector) by averaging the BS prices

        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        '''

        # Calculate the forward
        div_fac = np.exp(-texp*self.divr)
        disc_fac = np.exp(-texp*self.intr)
        forward = spot / disc_fac * div_fac
        sigma_0 = self.sigma / spot * forward  # adjust \sigma_0 accordingly

        # Generate the paths of sigma_t and normalized integrated variance
        vol_path = sigma_0 * self.sigma_path(texp)  # the path of sigma_t
        sigma_t = vol_path[-1, :]  # sigma_t at maturity (t=T)
        I_t = self.intvar_normalized(vol_path) / (sigma_0 ** 2 * texp) * texp

        # Calculate the equivalent spot and volatility of the Bachelier model
        vol = sigma_0 * np.sqrt((1 - self.rho**2) * I_t)  # Just an example
        spot_equiv = forward + self.rho * (sigma_t - sigma_0) / self.vov

        # Calculate the option prices
        m = self.base_model(vol)
        p = np.mean(m.price(strike[:, None], spot_equiv, texp, cp), axis=1)
        
        return p