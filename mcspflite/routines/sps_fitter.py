#!/usr/bin/env python

import numpy as np
import os, warnings
import string, gc, time
from scipy.special import erf
from astropy.modeling import models,fitting
import astropy.io.fits as fits
from scipy.interpolate import RectBivariateSpline as rect
from bisect import bisect_left
import readfilt
import astropy.units as u
import sys
from contextlib import contextmanager

from ..utils.magtools import getmag_spec

warnings.filterwarnings("ignore")

@contextmanager
def redir_stdout(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different



class sps_fitter:

    def __init__(self, redshift, phot_mod_file, flux_obs, eflux_obs, filter_list, lim_obs, \
            fesc=0., Gpriors=None, modeldir='./', filtdir='./', dl=None, cosmo=None):
        
        """ Class for dealing with MultiNest fitting """
                
        if cosmo is None:
           from astropy.cosmology import FlatLambdaCDM
           cosmo = FlatLambdaCDM(H0=70,Om0=0.3)
        
       #input information of the galaxy to be fitted
        self.redshift = redshift
        self.Gpriors = Gpriors
        self.modeldir = modeldir
        self.filtdir = filtdir
        
        #constants
        self.small_num = 1e-70
        self.lsun = 3.839e33 #erg/s
        self.pc2cm = 3.08568e18 #pc to cm
        self.clight  = 2.997925e18 #A/s
        self.h = 6.6262E-27 # Planck constant in erg s
        self.Mpc_to_cm = u.Mpc.to(u.cm)
        self.angstrom_to_km = u.angstrom.to(u.km)
        
        #Derive luminosity distance in Mpc
        if dl is None:
           dl = cosmo.luminosity_distance(redshift).value
        
        self.mag2cgs = np.log10(self.lsun/4.0/np.pi/(self.pc2cm*self.pc2cm)/100.)
        self.dist = 4.0*np.pi*(self.pc2cm*self.pc2cm)*100.
        self.lum_corr = (self.Mpc_to_cm*dl)**2*4.*np.pi #4pi*dl^2 in cm^2
        
        #some derived parameters
        self.gal_age = cosmo.age(self.redshift).value
        self.dm = 5.*np.log10(dl*1e5) #DM brings source at 10pc
        self.fscale = 10**(-0.4*self.dm)

        ## read in the pre-computed SPS grid
        
        mfile = fits.open(phot_mod_file)
        num_ext = len(mfile)
        
        #pull wavelength information from the primary extension of the models
        wdata = np.array(mfile[0].data, dtype=np.float)
        twl = self.airtovac(wdata)
        
        #first pass through extensions to get grid parameters
        ext_tau, ext_age, ext_metal = {}, {}, {}
        for ii in xrange(1,num_ext):
            age = mfile[ii].header['AGE'] / 1e3
            tau   = mfile[ii].header['TAU'] / 1e3
            ext_tau[ii]   = np.float(tau)
            ext_age[ii]   = np.float(age)

  
        #get the wavelength informmation for the DH02 templates to stitch the two together
        dh_wl = np.loadtxt(modeldir+'spectra_DH02.dat', usecols=(0,))*1e4
        dh_nwl = len(dh_wl)

        #expand wavelength grid to include range covered by DH templates
        dh_long = (dh_wl > twl.max())
        self.wl = np.r_[twl, dh_wl[dh_long]]
        self.n_wl = len(self.wl)

        self.grid_tau = np.unique(ext_tau.values())
        self.grid_age = np.unique(ext_age.values())
        
        self.n_tau = len(self.grid_tau)
        self.n_age = len(self.grid_age)
        
        tau_id = dict(zip(self.grid_tau, range(self.n_tau)))
        age_id = dict(zip(self.grid_age, range(self.n_age)))

        ### for nebular emission ###
        self.wl_lyman = 912.
        self.ilyman = np.searchsorted(self.wl, self.wl_lyman, side='left') #wavelength just above Lyman limit
        self.lycont_wls = np.r_[self.wl[:self.ilyman], np.array([self.wl_lyman])]
        self.clyman_young = [None, None] #A list of two elements, first is for phot, the other for spec
        self.clyman_old   = [None, None] #A list of two elements, first is for phot, the other for spec
        self.fesc = fesc                 #lyman continuum escape fraction
        
        #output grid
        self.mod_grid = np.zeros((self.n_tau, self.n_age, self.n_wl), dtype=np.float)
        
        #grid where the fractional flux from young populations is stored
        self.fym_grid = np.zeros_like(self.mod_grid)
        
        for ii in xrange(1,num_ext):
            tau_idx = tau_id[ext_tau[ii]]
            age_idx = age_id[ext_age[ii]]
            
            mdata = np.array(mfile[ii].data,  dtype=np.float)
            mmass = mfile[ii].header['MSTAR']
            metal = mfile[ii].header['METAL']
            
            self.mod_grid[tau_idx, age_idx, :] = np.interp(self.wl, twl, mdata[:, 0]/mmass, left=0, right=0)
            self.fym_grid[tau_idx, age_idx, :] = np.interp(self.wl, twl, mdata[:, 1], left=0, right=0)

        mfile.close() 

        #some information on the BC03 spectral resolution (in the optical)
        self.sps_res_val = np.copy(self.wl)/300.
        self.sps_res_val[(self.wl >= 3200) & (self.wl <= 9500)] = 3.
        
        self.mod_grid[~np.isfinite(self.mod_grid)] = 0.
        self.fym_grid[~np.isfinite(self.fym_grid)] = 0.

        #pre-compute attenuation curve
        self.k_cal = self._make_dusty(self.wl)
        
        #redshift the grid to be used in deriving predicted fluxes, also apply
        #flux correction to conserve energy
        self._redshift_spec()

        #set up for nebular emission lines
        self.emm_scales = np.zeros((7,10,128), dtype=np.float)
        self.emm_wls    = np.zeros(128,        dtype=np.float)
        self.emm_ages   = np.zeros(10,         dtype=np.float)
        self.emm_ions    = np.zeros(7,         dtype=np.float)
        icnt = 0
        rline = 0
        iline = 0
        with open(modeldir+'nebular_Byler.lines','r') as file:
            for line in file:
                if line[0] != '#':
                    temp = string.strip(line).split(None)
                    if not iline: #Read wave line
                        self.emm_wls[:] = np.array(map(float, temp))
                        iline = 1
                    else:
                        if rline: #Read line fluxes
                            self.emm_scales[icnt%7,icnt/7,:] = np.array(map(float, temp))
                            icnt += 1
                        if len(temp) == 3 and float(temp[0]) == 0.0:
                            rline = 1
                            self.emm_ages[icnt/7] = float(temp[1])/1e6
                            self.emm_ions[icnt%7]  = float(temp[2])
                        else:
                            rline = 0
        
        thb = (self.emm_wls > 4860) & (self.emm_wls < 4864)
        tha = (self.emm_wls > 6560) & (self.emm_wls < 6565)
        self.emm_scales = np.copy(self.emm_scales) / self.emm_scales[:,:,thb]
        
        mscale     = np.max(self.emm_scales, axis=(0,1))
        keep_scale = (mscale > 0.025) & (self.emm_wls<1E5)
                
        self.emm_scales = self.emm_scales[:,:,keep_scale]
        self.emm_wls    = self.emm_wls[keep_scale]
        
        #generate pattern arrays for nebular emission lines
        dpix = np.diff(self.wl)
        self.wl_edges  = np.r_[np.array([self.wl[0]-dpix[0]/2.]), np.r_[self.wl[1:]-dpix/2., np.array([self.wl[-1]+dpix[-1]/2.])]]
        self.res_lines = np.interp(self.emm_wls, self.wl, self.sps_res_val)/2.355
      
        self.emm_lines_all = np.zeros((len(self.emm_ions), len(self.emm_ages), len(self.wl)), dtype=np.float)
        for jj in range(len(self.emm_ions)):
          for ii in range(len(self.emm_ages)):
            this_scale = self.emm_scales[jj,ii,:]
            self.emm_lines_all[jj,ii,:] = np.sum(this_scale[:,None]*\
                    np.diff(0.5*(1.+erf((self.wl_edges[None,:]-self.emm_wls[:,None])/\
                    np.sqrt(2.*self.res_lines**2)[:,None])), axis=1)/np.diff(self.wl_edges), axis=0)
                
        #### LOAD DUST EMISSION TABLES ####
        #first fetch alpha values
        self.dh_alpha = np.loadtxt(modeldir+'alpha_DH02.dat', usecols=(0,))
        self.dh_nalpha = len(self.dh_alpha)

        self.dh_dustemm = np.zeros((self.dh_nalpha, self.n_wl), dtype=np.float)
        for ii in range(self.dh_nalpha):
            tdust = 10**np.loadtxt(modeldir+'spectra_DH02.dat', usecols=(ii+1,))
            self.dh_dustemm[ii,:] = np.interp(self.wl, dh_wl, tdust)/self.wl

        #normalize to Lbol = 1
        norm = np.trapz(self.dh_dustemm, self.wl)
        self.dh_dustemm /= norm[:,None]
        
        #### LOAD DUST EMISSION TABLES ####        
                
        self.filters = filter_list #should already correspond to FSPS names
        self.n_bands = len(self.filters)
        self.bands, self.pivot_wl = self._get_filters()
        
        #photometric measurements (convert from fnu to flambda)
        #Input at this stage is in erg/cm^2/s/Hz output in erg/cm2/s/A
        self.flux_obs = flux_obs * self.clight/self.pivot_wl**2
        self.eflux_obs = eflux_obs * self.clight/self.pivot_wl**2
        self.lim_obs = lim_obs
        
        #set up parameter limits
        self.tau_lims = np.array((self.grid_tau.min(), self.grid_tau.max()))
        self.age_lims = np.array((self.grid_age.min(), self.gal_age)) #self.grid_age.max()))
        self.av_lims = np.array((0., 4.))
        self.alpha_lims = np.array((self.dh_alpha[0], self.dh_alpha[-1]))
        self.mass_lims = np.array((7,12))
        self.emmage_lims = np.array((self.emm_ages.min(), 10))
        self.emmion_lims = np.array((self.emm_ions.min(), self.emm_ions.max()))

        self.bounds = [self.tau_lims, self.age_lims, self.av_lims, \
                self.alpha_lims, self.mass_lims, self.emmage_lims, self.emmion_lims]
                
        self.ndims = len(self.bounds)   

    def vactoair(self, linLam):
        """Convert vacuum wavelengths to air wavelengths using the conversion
        given by Morton (1991, ApJS, 77, 119).

        """
        wave2 = np.asarray(linLam, dtype=float)**2
        fact = 1. + 2.735182e-4 + 131.4182/wave2 + 2.76249e8/(wave2*wave2)
        return linLam/fact


    def airtovac(self, linLam):
        """Convert air wavelengths to vacuum wavelengths using the conversion
        given by Morton (1991, ApJS, 77, 119).

        """
        sigma2 = np.asarray(1E4/linLam, dtype=float)**2
        
        fact = 1. + 6.4328e-5 + 2.949281e-2/(146.-sigma2) + 2.5540e-4/(41.-sigma2)
        fact[linLam < 2000] = 1.0
        
        return linLam*fact
    
    def _scale_cube(self, cube, ndims, nparams):
        for ii in range(ndims):
            cube[ii] = cube[ii]*self.bounds[ii].ptp() + np.min(self.bounds[ii])

        return


    def _make_dusty(self, wl):
        
        #compute attenuation assuming Calzetti+ 2000 law
        #single component 
        n_wl = len(wl)
        R = 4.05
        div = wl.searchsorted(6300., side='left')
        k_cal = np.zeros(n_wl, dtype=float)
        
        k_cal[div:] = 2.659*( -1.857 + 1.04*(1e4/wl[div:])) + R
        k_cal[:div] = 2.659*(-2.156 + 1.509*(1e4/wl[:div]) - 0.198*(1e4/wl[:div])**2 + 0.011*(1e4/wl[:div])**3) + R
        
        zero = bisect_left(-k_cal, 0.)
        k_cal[zero:] = 0.

        #2175A bump
        #eb = 1.0
        k_bump = np.zeros(n_wl, dtype=float)
        #k_bump[:] = eb*(wl*350)**2 / ((wl**2 - 2175.**2)**2 + (wl*350)**2)
        
        #k_tot is the total selective attenuation A(lam)/E(B-V).
        #For calzetti R(V) = A(V)/E(B-V)
        k_tot = k_cal + k_bump 
                
        #Return 0.4*A(lam)/A(V)
        return 0.4*(k_cal+ k_bump)/R

    def _redshift_spec(self):
        self.red_wl       = self.wl * (1.+self.redshift)
        if self.redshift>0:
           self.red_mod_grid = self.mod_grid / (1+self.redshift)  
        else:
           self.red_mod_grid = np.copy(self.mod_grid)
 
    def _tri_interp(self, data_cube, value1, value2, value3, array1, array2, array3):
        #locate vertices
        ilo = bisect_left(array1, value1)-1
        jlo = bisect_left(array2, value2)-1
        klo = bisect_left(array3, value3)-1

        di = (value1 - array1[ilo])/(array1[ilo+1]-array1[ilo])
        dj = (value2 - array2[jlo])/(array2[jlo+1]-array2[jlo])
        dk = (value3 - array3[klo])/(array3[klo+1]-array3[klo])

        interp_out = data_cube[ilo,jlo,klo,:]       * (1.-di)*(1.-dj)*(1.-dk) + \
                     data_cube[ilo,jlo,klo+1,:]     * (1.-di)*(1.-dj)*dk + \
                     data_cube[ilo,jlo+1,klo,:]     * (1.-di)*dj*(1.-dk) + \
                     data_cube[ilo,jlo+1,klo+1,:]   * (1.-di)*dj*dk + \
                     data_cube[ilo+1,jlo,klo,:]     * di*(1.-dj)*(1.-dk) + \
                     data_cube[ilo+1,jlo,klo+1,:]   * di*(1.-dj)*dk + \
                     data_cube[ilo+1,jlo+1,klo,:]   * di*dj*(1.-dk) + \
                     data_cube[ilo+1,jlo+1,klo+1,:] * di*dj*dk

        return interp_out

    def _bi_interp(self, data_cube, value1, value2, array1, array2):
        #locate vertices
        ilo = bisect_left(array1, value1)-1
        jlo = bisect_left(array2, value2)-1

        di = (value1 - array1[ilo])/(array1[ilo+1]-array1[ilo])
        dj = (value2 - array2[jlo])/(array2[jlo+1]-array2[jlo])

        interp_out = data_cube[ilo,jlo,:]     * (1.-di)*(1.-dj) + \
                     data_cube[ilo,jlo+1,:]   * (1.-di)*dj + \
                     data_cube[ilo+1,jlo,:]   * di*(1.-dj) + \
                     data_cube[ilo+1,jlo+1,:] * di*dj

        return interp_out

    def _interp(self, data_cube, value, array):
        #locate vertices
        ilo = bisect_left(array, value)-1
        di = (value - array[ilo])/(array[ilo+1]-array[ilo])

        interp_out = data_cube[ilo,:]   * (1.-di) + \
                     data_cube[ilo+1,:] * di

        return interp_out


    def _get_filters(self):
        #fetch filter transmission curves from FSPS
        #normalize and interpolate onto standard grid
        bands = np.zeros((self.n_bands, self.n_wl), dtype=float)
        pivot = np.zeros(self.n_bands, dtype=float)
        
        #lookup for filter number given name
        filters_db = readfilt.init_filters(self.filtdir)
        
        for ii, filt in enumerate(self.filters):
            
            if 'line' in filt:
             return 0,0
           
            fobj = readfilt.get_filter(filters_db, filt)
            fwl, ftrans = fobj.transmission
            ftrans = np.maximum(ftrans, 0.)
            trans_interp = np.asarray(np.interp(self.red_wl, fwl, \
                    ftrans, left=0., right=0.), dtype=np.float) 

            #normalize transmission
            ttrans = np.trapz(np.copy(trans_interp)*self.red_wl, self.red_wl) #for integrating f_lambda
            if ttrans < self.small_num: ttrans = 1.
            ntrans = np.maximum(trans_interp / ttrans, 0.0)
            
            if 'mips' in filt:
                td = np.trapz(((fobj.lambda_eff/self.red_wl)**(2.))*ntrans*self.red_wl, self.red_wl)
                ntrans = ntrans/max(1e-70,td)

            if 'irac' in filt or 'pacs' in filt or 'spire' in filt or 'iras' in filt: 
                td = np.trapz(((fobj.lambda_eff/self.red_wl)**(1.))*ntrans*self.red_wl, self.red_wl)
                ntrans = ntrans/max(1e-70,td)

            bands[ii,:] = ntrans
            pivot[ii] = fobj.lambda_eff
        
        return bands, pivot

    def _get_mag_single(self, spec, ret_flux=True):
        
        #compute observed frame magnitudes and fluxes, return both
        
        tflux = np.zeros(self.n_bands, dtype=np.float)

        getmag_spec(self.red_wl, np.einsum('ji,i->ij', self.bands, \
                spec*self.red_wl), self.n_bands, tflux)
        
        if not ret_flux:
            tmag = -2.5*np.log10(tflux*self.fscale) - 48.6
            if np.all(tflux) > 0:
                return tmag
            else:
                tmag[flux <= 0] = -99.
                return tmag
        
        #Return fluxes in erg/s/cm^2/A
        if np.all(tflux) > 0:        
            return tflux*self.fscale
        else:
            flux = tflux*self.fscale
            flux[tflux <= 0] = 0.
            return flux 

    def lnprior(self, p, ndim):
        if all(b[0] <= v <= b[1] for v, b in zip(p, self.bounds)):
            
            pav = 0
            
            if self.Gpriors is not None:
              for par in range(ndim):
                if self.Gpriors[2*par] != 'none' and self.Gpriors[(2*par)+1] != 'none':
                  val = float(self.Gpriors[2*par])
                  sig = float(self.Gpriors[(2*par)+1])
                  pav  +=  -0.5*(((p[par]-val)/sig)**2 + np.log(2.*np.pi*sig**2))
                            
            return pav

        return -np.inf

    def lnlhood(self, p, ndim, nparams):
        
           
        model_phot, _ = self.reconstruct_phot(p, ndim)
         
        if np.all(model_phot == 0.):
           return -np.inf

        iphot2 = 1./(self.eflux_obs**2)
         
        if np.sum(self.lim_obs):
             terf = 0.5*(1.+erf((self.eflux_obs-model_phot)/np.sqrt(2.)/self.eflux_obs))[self.lim_obs == 1]
             if np.any(terf == 0):
                 return -np.inf
             else:
                 phot_lhood = np.nansum(-0.5*((iphot2*(self.flux_obs-model_phot)**2)[self.lim_obs == 0] - \
                         np.log(iphot2[self.lim_obs == 0]) + np.log(2.*np.pi))) + \
                         np.nansum(np.log(terf))
        else:
             phot_lhood = np.nansum(-0.5*((iphot2*(self.flux_obs-model_phot)**2) - \
                     np.log(iphot2) + np.log(2.*np.pi)))
        
        #### APPLY THE PRIOR HERE  #####
        pr = self.lnprior(p, ndim)
        
        if not np.isfinite(pr):
            return -np.inf
        
        return phot_lhood + pr

        
    def _get_clyman(self, spec): #compute number of Lyman continuum photons
        lycont_spec = np.interp(self.lycont_wls, self.wl, spec) #spectrum in erg/s/A
        nlyman = np.trapz(lycont_spec*self.lycont_wls, self.lycont_wls)/self.h/self.clight
 
        #modify input spectrum to remove photons 
        spec[:self.ilyman] *= self.fesc
    
        return nlyman*(1.-self.fesc), spec
      
    def _get_nebular(self, emm_spec, lyscale, index): 
        
        emm_young = self.clyman_young[index] * 4.796e-13 * emm_spec * lyscale #conversion is from QH0 to Hbeta luminosity
        emm_old   = self.clyman_old[index]   * 4.796e-13 * emm_spec * lyscale
                
        return emm_young, emm_old

    def _make_spec_emm(self, vel, sigma, emm_age, emm_ion):
        vel_pix = vel/self.kms2pix
        sigma_pix = sigma/self.kms2pix

        temp_emm_scales = self._bi_interp(self.log_emm_scales, emm_ion, emm_age, self.emm_ions, self.emm_ages) 
                
        emm_grid = np.sum(temp_emm_scales[:,None]*\
                np.diff(0.5*(1.+erf((self.diff_pix-vel_pix-self.vsys_pix)/np.sqrt(2.)/sigma_pix)), axis=1)/\
                np.diff(10**self.log_model_wl_edges)[None,:], axis=0)

        return emm_grid


    def reconstruct_phot(self, p, ndim):
        #parameters
        itau, iage, iav, ialpha, ilmass, iage_gas, iion_gas = [p[x] for x in range(ndim)]
        
        #interpolate the full photometric grid
        spec_model = self._bi_interp(self.red_mod_grid, itau, iage, self.grid_tau, self.grid_age)
        frac_model = self._bi_interp(self.fym_grid,     itau, iage, self.grid_tau, self.grid_age)

        #get number of lyman continuum photons
        self.clyman_young[0], temp_young = self._get_clyman(spec_model*frac_model)
        self.clyman_old[0], temp_old     = self._get_clyman(spec_model*(1.-frac_model))

        #### Include nebular emission ####
        iemm_lines = self._bi_interp(self.emm_lines_all, iion_gas, iage_gas, self.emm_ions, self.emm_ages) 

        emm_young, emm_old = self._get_nebular(iemm_lines, 1, 0)

        #attenuate photometry spectrum, then compute fluxes given input bands
        self.dusty_phot_young = (10**(-(2.27*iav)*self.k_cal) * (temp_young + emm_young))
        self.dusty_phot_old   = (10**(-iav*self.k_cal) * (temp_old+emm_old))
        self.dusty_phot       = self.dusty_phot_young + self.dusty_phot_old
        
        #### THERMAL DUST EMISSION ####
        lbol_init = np.trapz(temp_young+temp_old+emm_young+emm_old, self.wl)
        lbol_att = np.trapz(self.dusty_phot, self.wl)

        dust_emm = (lbol_init - lbol_att)
        tdust_phot = self._interp(self.dh_dustemm, ialpha, self.dh_alpha)

        #remove stellar component
        mask_pixels = (self.wl >= 2.5e4) & (self.wl <= 3e4)
        scale = np.sum(spec_model[mask_pixels]*tdust_phot[mask_pixels]) / np.sum(spec_model[mask_pixels]*spec_model[mask_pixels])
        
        tdust_phot -= scale*spec_model
        tdust_phot[(self.wl < 2.5e4) | (tdust_phot < 0.)] = 0.
        
        norm = np.trapz(tdust_phot, self.wl) 

        dust_phot = np.copy(tdust_phot) * dust_emm / norm
        tdust_phot = 0.
        icnt = 0.
        lboln, lbolo = 0, 1e5
        
        while (lbolo-lboln) > 1e-15 and icnt < 5:
            idust_phot = np.copy(dust_phot)
            dust_phot = dust_phot * 10**(-iav*self.k_cal)

            tdust_phot += dust_phot
            
            lboln = np.trapz(dust_phot, self.wl)
            lbolo = np.trapz(idust_phot, self.wl)
            dust_phot = np.maximum(tdust_phot*(lbolo-lboln)/norm, self.small_num) 
            icnt += 1

        self.dusty_phot_dust = tdust_phot
        flux_model = self._get_mag_single(self.dusty_phot+tdust_phot)

        return flux_model*(10**ilmass), (self.dusty_phot+tdust_phot)*(10**ilmass)
    
    def __call__(self, p):
        lp = self.lnprior(p, ndim)
        if not np.isfinite(lp):
            return -np.inf

        lh = self.lnlhood(p)
        if not np.isfinite(lh):
            return -np.inf

        return lh + lp

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, trace):
        gc.collect()


