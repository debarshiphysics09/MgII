import matplotlib
matplotlib.use('Agg')

from astropy.io import fits as pyfits
import numpy as np
from importlib import import_module
from extinction import fitzpatrick99 as f99
from extinction import remove
from lmfit import Model
from lmfit.models import GaussianModel, LorentzianModel
from astropy.convolution import convolve, Box1DKernel
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
import sys,os,time,datetime
import signal,gc
sys.path.insert(0, './params_alt')


#==============================================
#Define the fitting functions used for lmfit (continuum and Fe-template); adapted/copied from other code
def powerLaw(x, amp, exp):
	"""Model for powerlaw"""
	model = amp*(x/5100.0)**(exp)
	return model

Fe_in = np.genfromtxt('Vestergaard.txt').T
Fe_temp = interp1d(Fe_in[0],Fe_in[1],fill_value='extrapolate')
del Fe_in

def FeUV(x, redshift, amp, sigma,limit=(2000,3089)):
	"""Final output for FeII template, smoothed with gaussian filter"""
	for ii in range(len(x)):
		if x[ii]/(1.+redshift) > limit[0]: break
		else: np.delete(x,ii)
	for ii in reversed(range(len(x))):
		if x[ii]/(1.+redshift) < limit[1]: break
		else: np.delete(x,ii)
	Fez = amp*Fe_temp(x/(1.+redshift))
	model = gaussian_filter(Fez, sigma)
	return model
#==============================================


class SpectrumError(Exception):
	""""Used to control flow in loops that call functions from the Spectrum class, as well as in the Spectrum method 'choose_model'. Necessary due to possible error in lmfit chi2 fitting, called in the Spectrum method 'run_fit' """
	pass

class Spectrum:
	"""Spectral data for a single epoch, which will be adjusted according to airmass for BOSS spectra and according to OIII scaling for WHT/MMT/Magellan scaling. The adjusted spectrum will subsequently be fitted using the lmfit routine

		attributes:
		data: the spectrum to be loaded
		data_fit: the trimmed spectrum (note that 'data' will store the untrimmed spectrum, which can be re-used by trimming it for fits to different ranges)
		airmass: calculated from the spectral fits file for BOSS spectra
		correction: taken from a pre-calculated file
		output: array to contain data to write to text file
		z: redshift
		A_v: visual extinction
		lower: lower bound on spectral range
		upper: upper bound on spectral range
		fe_yn: marker to indicate whether or not to include the Fe-template in the fitting process
		cmod: the model defined for lmfit
		pars: the parameters of cmod, defined for lmfit
		fit_out: results of the fit
		mod: module to be loaded, which included the predefined initial paraments for the fitting process
		survey: the survey to which the current spectrum belongs (important for spectral corrections)
		corr_min & corr_max: lower and upper bounds on the correction function for BOSS spectra
		Fe_template: the loaded FeII-template
		final_pars: dictionary for the fitted values from the last iteration of the fit for each included parameter
		final_fits: dictionary containing the best-fit functions (matching the function parameters in final_pars)
"""

	def __init__(self, full_name, data=None, airmass=0, correction=1, output=None, z=1, A_v=0, lower=None, upper=None,fe_yn='no',Fe_template='Vestergaard.txt',path=None,fitspath=None,Spath=None,failed_runs='failedrunslist'):
		self.full_name = full_name
		self.obs_date = self.full_name[8:]
		self.obj = self.full_name[:7]
		self.data = data
		self.data_fit = None
		self.airmass = airmass
		self.correction = correction
		self.output = output
		self.z = z
		self.A_v = A_v
		self.lower = lower
		self.upper = upper
		self.fe_yn = fe_yn
		self.cmod = None
		self.pars = None
		self.fit_out = None
		self.mod = None
		self.survey = None
		self.path = path
		self.Spath = Spath
		self.fitspath = fitspath
		self.failed_runs = failed_runs
		self.final_pars = {}
		self.final_fits = {}

		self.line = 'UV' #this definition is for labelling only


	def __del__(self):
		gc.collect()
		print ('Finished with spectrum for {}'.format(self.full_name))


	def load(self):
		"""Load the spectral data into a numpy array. The input data can be either a text file (format = wavelength, flux, flux_err), or an SDSS formatted fits file"""
		if self.full_name.endswith('.fits'):
			hdu_in = pyfits.open(self.Spath+self.full_name)
			tb = hdu_in[1].data
			hdu_in.close()
			lam = tb.field('loglam')
			base = np.full(len(lam),10)
			lam = np.power(base,lam)
			flux = tb.field(0)*1e-17
			flux_err = tb.field('ivar')
			id_z = np.where(flux_err==0)
			lam = np.delete(lam,id_z); flux = np.delete(flux,id_z); flux_err = np.delete(flux_err,id_z)
			flux_err = (1/np.sqrt(flux_err))*1e-17
			self.data = np.array([lam,flux,flux_err]).T
			self.obs_date = str(self.full_name[10:15])
		else:
			if os.path.isfile(self.Spath+'sdss_'+self.full_name):
				self.mod = import_module('sdss_'+self.obj)
				self.data = np.loadtxt(self.Spath+'sdss_'+self.full_name)
				self.survey = 'sdss'
			else:
				try:
					self.data = np.loadtxt(self.Spath+self.full_name)
					self.mod = import_module(self.obj)
				except Exception:
					print 'Error: could not load data for {}'.format(self.full_name)
			self.z = self.mod.z_est; self.A_v = self.mod.A_v; self.lower = self.mod.lower; self.upper = self.mod.upper; self.fe_yn = self.mod.fev

	def load_corrected_redshift(self):
		'''In case a clearly erroneous redshift is loaded as an initial estimate, try to solve the issue by loading the redshift from the SDSS spectrum file itself'''
		hdu_in = pyfits.open(self.Spath+self.full_name)
		try:
			if hdu_in[2].data['z'] > 0:
				self.z = hdu_in[2].data['z']
			else:
				self.z = np.average( hdu_in[3].data['linez'][np.where(hdu_in[3].data['linez'] > 0)] )
		except Exception:
			self.z = 0

	def adjust_boss_spectra(self,airmass=None):
		"""Apply the correction from Harris et al. to the BOSS spectra. Use the preloaded correction function, and interpolate for the bins if necessary. The airmass used for the correction function is the average airmass over the period of observation, calculated by taking the mean of the airmasses indicated for the indivdual exposures in the SDSS spectrum files, weighted by the exposure times. The spectrum is trimmed to match the BOSS correction function limits, if necessary"""
		if self.survey=='sdss' and int(self.obs_date)>55000: #The MJD is selected to be in between the SDSS I/II and SDSS III epochs
			for ii in reversed(range(len(self.data))):
				if self.data[ii,0] < self.corr_min or self.data[ii,0] > self.corr_max:
					self.data = np.delete(self.data,ii,0)
			if airmass:
				self.airmass = airmass
			else:
				airmass_list = []; exposure_times = []
				hdu_list = pyfits.open(self.fitspath+self.obj+'/sdss_'+self.obj+'_'+self.obs_date+'.fits')
				for ii in range(4,len(hdu_list)):
					if hdu_list[ii].header['EXTNAME'][0]=='B':
						airmass_list += [hdu_list[ii].header['AIRMASS']]
						exposure_times += [hdu_list[ii].header['EXPTIME']]
				hdu_list.close()
				airmass_list = np.array(airmass_list); exposure_times = np.array(exposure_times)
				self.airmass = np.average(airmass_list,weights=exposure_times)
			self.data[:,1] = self.data[:,1]*(self.corr_slope(self.data[:,0])*self.airmass+self.corr_intercept(self.data[:,0]))
			self.data[:,2] = self.data[:,2]*abs(self.corr_slope(self.data[:,0])*self.airmass)

	def adjust_new_spectra(self,scaling=[]):
		"""Apply the correction factors (from Chelsea) based on assumed constancy of flux in the narrow OIII line. These corrections apply to the WHT, Magellan and MMT spectra"""
		if self.survey != 'sdss' and self.obj in scaling['object']:
			self.data[:,1:] = self.data[:,1:]*scaling[np.where(scaling['object']==self.obj)][0][1]

	def deredden_CCM(self,R_v=3.1):
		"""Deredden the spectrum, based on the estimated A_v, using the Cardelli, Claython adn Mathis (1989) extinction curve"""
		#Calculate the dereddening factor for every wavelength bin
		factorlist = []
		for ii in self.data[:,0]:
			y = 1./(ii*10**-4) - 1.82
			a = 1. + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
			b =     1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
			CCM = (a + b/R_v)*self.A_v
			factor = 10**(CCM/2.5)
			factorlist += [factor]
		factorlist = np.array(factorlist)
		#Apply the dereddening factors
		self.data[:,1] = self.data[:,1]*factorlist
		self.data[:,2] = self.data[:,2]*factorlist

	def deredden(self,R_v=3.1):
		"""Deredden the spectrum, based on the estimated A_v, using the Fitzpatrick (1999) curve, as used in Schlafy & Finkbeiner (2012)"""
		self.data = self.data.T
		self.data[1] = remove(f99(self.data[0],self.A_v,R_v),self.data[1])
		self.data[2] = remove(f99(self.data[0],self.A_v,R_v),self.data[2])
		self.data = self.data.T

	def shift_mc(self):
		'''Create a new spectrum by adding 1-sigma Gaussion noise to the input spectrum'''
		if len(self.data)!=3:
			self.data=self.data.T
		smoothed_err = np.copy(self.data[2]); avg_err = np.average(self.data[2])
		for ii in range(len(smoothed_err)):
			if smoothed_err[ii]>2*avg_err: smoothed_err[ii]=avg_err
		self.data[1] = np.random.normal(self.data[1],smoothed_err,len(self.data[1]))
		self.data=self.data.T

	def trim_to_range(self, line='yes',continuum=None):
		"""Trim wavelength range according to the lower and upper bounds specified in the parameter file, when trimming for the line fit. Trim to the other specified ranges to prepare the spectrum for a fit to the continuum. This method loads the original data to be fitted, and can therefore be used to restart the fitting process after iterative fitting has altered the previous copy of data_fit. In case the resultin data has two few points for a reasonable fit, the function will raise a SpectrumError exception"""
		if self.z < 0: self.load_corrected_redshift()
		self.data_fit = np.copy(self.data)
		if continuum=='mgii':
			for ii in reversed(range(len(self.data))):
				ll = self.data[ii,0]
				sz = 1.+self.z
#				if ll<sz*1445 or (ll>sz*1465 and ll<sz*1700) or (ll>sz*1705 and ll<sz*2200) or (ll>sz*2700 and ll<sz*2900) or ll>sz*3088:
				if ll<sz*2300 or (ll>sz*2700 and ll<sz*2900) or ll>sz*3088:
					self.data_fit = np.delete(self.data_fit,ii,0)
		elif continuum=='hbeta_oiii':
			for ii in reversed(range(len(self.data))):
				ll = self.data[ii,0]
				sz = 1.+self.z
				if ll<sz*4500 or ll>sz*5300:
					self.data_fit = np.delete(self.data_fit,ii,0)
		elif continuum=='oii':
			for ii in reversed(range(len(self.data))):
				ll = self.data[ii,0]
				sz = 1.+self.z
				if ll<sz*3600 or ll>sz*3800:
					self.data_fit = np.delete(self.data_fit,ii,0)
		elif line=='yes':
			for ii in reversed(range(len(self.data))):
				if self.data[ii,0] < self.lower or self.data[ii,0] > self.upper:
					self.data_fit = np.delete(self.data_fit,ii,0)
		if len(self.data_fit) < 10:
			with open(self.failed_runs,'a') as of:
				of.write(self.full_name+'\n')
			raise SpectrumError('Error for {}, no data in fitting range'.format(self.full_name))

	def mask(self,x_lim):
		'''Mask an area of the spectrum. The range in defined by x_lim.'''
		del_rng = []
		for ii in range(len(self.data_fit)):
			if self.data_fit.T[0][ii]>(x_lim[0]*(1+self.z)) and self.data_fit.T[0][ii]<(x_lim[1]*(1+self.z)):
				del_rng += [ii]
		self.data_fit = np.delete(self.data_fit,del_rng,0)
		if len(self.data_fit) < 10:
			with open(self.failed_runs,'a') as of:
				of.write(self.full_name+'\n')
			raise SpectrumError('Error for {}, no data in fitting range'.format(self.full_name))

	def set_models(self,continuum_only='no',continuum_feii_only='no',line_only='no'):
		"""Define the models to be used in the lmfit fitting process"""
		if continuum_only == 'yes':
			self.cmod = ( Model(powerLaw,prefix='pow'+self.line) )
		elif continuum_feii_only=='yes':
			self.cmod = ( Model(powerLaw,prefix='pow'+self.line)
					+ Model(FeUV, prefix='FeUV')
					)
		elif line_only=='yes':
			print 'only the lines'
		else:
			if self.fe_yn=='yes':
				self.cmod = ( Model(powerLaw, prefix='pow'+self.line)
						+ Model(FeUV, prefix='FeUV')
						+ GaussianModel(prefix='MgIIb')
						)
			else:
				self.cmod = ( Model(powerLaw, prefix='pow'+self.line)
						+ GaussianModel(prefix='MgIIb')
					)
		setHints = getattr(self.mod,'setHints_'+self.obs_date+'UV')
		setHints(self.cmod, self.z, self.lower, self.upper)
		self.pars = self.cmod.make_params()


	def set_models_iter(self,model = 'continuum_only'):
		'''Define the models to be used in the lmfit fitting process. The initial parameters for the fitting process are generic'''
		if model == 'continuum_only':
			self.cmod = ( Model(powerLaw,prefix='pow'+self.line) )
			self.cmod.set_param_hint('pow'+self.line+'amp',value=1e-17)
			self.cmod.set_param_hint('pow'+self.line+'exp',value=-1.0,max=0)
		elif model=='continuum_feii_only':
			self.cmod = ( Model(powerLaw,prefix='pow'+self.line)
					+ Model(FeUV, prefix='FeUV') )
			self.cmod.set_param_hint('pow'+self.line+'amp',value=1e-17)
			self.cmod.set_param_hint('pow'+self.line+'exp',value=-1.0,max=0)
			self.cmod.set_param_hint('Fe'+self.line+'amp',value=2e-3)
			self.cmod.set_param_hint('Fe'+self.line+'sigma',value=5,min=0)
			self.cmod.set_param_hint('Fe'+self.line+'redshift',value=self.z)
			if self.fit_out: # this line includes the previous continuum fit values in the fit
				for name in self.cmod.param_names:
					if name[-3:]!='Err' and name in self.fit_out.params:
						self.cmod.set_param_hint(name,value=self.fit_out.params[name].value)
			self.cmod.name = model
		elif model=='MgII':
			self.cmod = ( GaussianModel(prefix='MgII') )
			self.cmod.set_param_hint('MgIIcenter',value=2798*(1+self.z))
			self.cmod.set_param_hint('MgIIsigma',value=40,max=100)
			self.cmod.set_param_hint('MgIIamplitude',value=5e-15)
			self.lower, self.upper = 2700.*(1+self.z), 2900.*(1+self.z) #Values chosen for consistency with gaps in continuum fitting region
		elif model=='MgII_2':
			self.cmod = ( GaussianModel(prefix='MgII') + GaussianModel(prefix='MgIIw') )
			self.cmod.set_param_hint('MgIIcenter',value=2798*(1+self.z))
			self.cmod.set_param_hint('MgIIsigma',value=10,max=25)
			self.cmod.set_param_hint('MgIIamplitude',value=2e-15)
			self.cmod.set_param_hint('MgIIwcenter',value=2798*(1+self.z))
			self.cmod.set_param_hint('MgIIwsigma',value=40,max=65)
			self.cmod.set_param_hint('MgIIwamplitude',value=7e-15)
			self.lower, self.upper = 2700.*(1+self.z), 2900.*(1+self.z) #Values chosen for consistency with gaps in continuum fitting region
		elif model=='Hbeta_OIII':
			self.cmod = ( GaussianModel(prefix='Hbb') + GaussianModel(prefix='OIII_4959') + GaussianModel(prefix='OIII_5007') + GaussianModel(prefix='OIII_5007b') )
			self.cmod.set_param_hint('Hbbcenter', value = 4861*(1+self.z))
			self.cmod.set_param_hint('Hbbsigma',value = 35)
			self.cmod.set_param_hint('Hbbamplitude',value = 1e-15)
			self.cmod.set_param_hint('OIII_4959center',value = 4959*(1+self.z))
			self.cmod.set_param_hint('OIII_4959sigma',value = 2)
			self.cmod.set_param_hint('OIII_4959amplitude',value = 1e-15)
			self.cmod.set_param_hint('OIII_5007center',value = 5007*(1+self.z))
			self.cmod.set_param_hint('OIII_5007sigma',value = 2)
			self.cmod.set_param_hint('OIII_5007amplitude',value = 1e-16)
			self.cmod.set_param_hint('OIII_5007bcenter', expr = 'OIII_5007center', value = 5007*(1+self.z))
			self.cmod.set_param_hint('OIII_5007bsigma',value = 10)
			self.cmod.set_param_hint('OIII_5007bamplitude',value = 1e-16)
			self.lower, self.upper = 4820*(1.+self.z),5100*(1.+self.z)
		elif model=='OIII':
			self.cmod = ( GaussianModel(prefix='OIII_4959') + GaussianModel(prefix='OIII_5007'))
			self.cmod.set_param_hint('OIII_4959center',expr = 'OIII_5007center*(4959./5007.)',value = 4959*(1+self.z))
			self.cmod.set_param_hint('OIII_4959sigma',value = 2)
			self.cmod.set_param_hint('OIII_4959amplitude',value = 1e-15)
			self.cmod.set_param_hint('OIII_5007center', value = 5007*(1+self.z))
			self.cmod.set_param_hint('OIII_5007sigma',value = 10)
			self.cmod.set_param_hint('OIII_5007amplitude',value = 1e-16)
			self.lower, self.upper = 4910*(1.+self.z),5050*(1.+self.z)
		elif model=='OII':
			self.cmod = ( GaussianModel(prefix='OII_3727') )
			self.cmod.set_param_hint('OII_3727center',value = 3727*(1+self.z))
			self.cmod.set_param_hint('OII_3727sigma',value = 5)
			self.cmod.set_param_hint('OII_3727amplitude',value = 5e-15)
			self.lower, self.upper = 3700*(1+self.z),3750*(1+self.z)
		else: print 'No valid set of models selected in set_models_iter'
		self.pars = self.cmod.make_params()


	def use_fit_results(self,sigma_clip=5,limits={},expr={}):
		'''-1- Load results from a previous fit, and set these as the initial values of cmod\n-2- Sigma clip (5 sigma) the data with respect to the initial fit\nThese steps allow for iterative fitting. The method will use all the parameters which were included in the previous fit. The kwarg limits should be a dictionary: {'parname':(min,max), etc.}'''
		sig = np.std(self.fit_out.residual)
		for ii in reversed( range(len(self.data_fit[1])) ):
			if abs(self.fit_out.residual[ii])>=sigma_clip*sig:
				self.data_fit = np.delete(self.data_fit.T,ii,axis=0).T
		if len(self.data_fit.T) < 10:
			raise SpectrumError('Error for {}, no data in fitting range'.format(self.full_name))
		for name in self.cmod.param_names:
			if name[-3:]!='Err' and name in self.fit_out.params.keys():
				if name in expr:
					self.cmod.set_param_hint(name,value=self.fit_out.params[name].value,expr=expr[name])
				elif name in limits:
					self.cmod.set_param_hint(name,value=self.fit_out.params[name].value,min=limits[name][0],max=limits[name][1])
				else:
					self.cmod.set_param_hint(name,value=self.fit_out.params[name].value)
		self.pars = self.cmod.make_params()

	def time_handler(s,f):
		signal.alarm(0)
		raise SpectrumError('Timed out')
	signal.signal(signal.SIGALRM, time_handler)

	def run_fit(self):
		"""Run the lmfit algorithm, after the parameters have been set using the methods load and set_models"""
		if len(self.data_fit) != 3:
			self.data_fit = self.data_fit.T
		try:
			signal.alarm(15)
			self.fit_out = self.cmod.fit(self.data_fit[1], self.pars, x=self.data_fit[0], weights=np.sqrt(1./self.data_fit[2]**2.),fit_kws={'maxfev':100})
		except ValueError:
			with open(self.failed_runs,'a') as of:
				of.write(self.full_name+'\n')
			print 'ValueError encountered, skipped fit for {}'.format(self.full_name)
			raise SpectrumError('Could not fit the data for {}'.format(self.full_name))
			return
		signal.alarm(0)
		for name in self.cmod.param_names:
			self.final_pars[name] = self.fit_out.params[name].value
			self.final_pars[name+'Err'] = self.fit_out.params[name].stderr
		for comp in self.cmod.components:
			if comp.prefix == 'pow'+self.line:
				rng = self.data.T[0][np.argwhere(self.data.T[0]==self.data_fit[0][0])[0][0]:np.argwhere(self.data.T[0]==self.data_fit[0][-1])[0][0]]
				if rng[0]>(2750*(1+self.z)):
					for ii in range(len(self.data.T[0])):
						if self.data.T[0][ii]>(2750*(1+self.z)):
							st_ind = ii
							break
					rng = self.data.T[0][st_ind:np.argwhere(self.data.T[0]==self.data_fit[0][-1])[0][0]]
				self.final_fits[comp.prefix] = (rng, powerLaw(rng,self.fit_out.params['powUVamp'].value,self.fit_out.params['powUVexp'].value))
			elif comp.prefix == 'FeUV':
				rng = self.data.T[0][np.argwhere(self.data.T[0]==self.data_fit[0][0])[0][0]:np.argwhere(self.data.T[0]==self.data_fit[0][-1])[0][0]]
				if rng[0]>(2750*(1+self.z)):
					for ii in range(len(self.data.T[0])):
						if self.data.T[0][ii]>(2750*(1+self.z)):
							st_ind = ii
							break
					rng = self.data.T[0][st_ind:np.argwhere(self.data.T[0]==self.data_fit[0][-1])[0][0]]
				self.final_fits[comp.prefix] = (rng, FeUV(rng,self.fit_out.params['FeUVredshift'].value,self.fit_out.params['FeUVamp'].value,self.fit_out.params['FeUVsigma'].value))
			else:
				self.final_fits[comp.prefix] = (self.data_fit[0],self.fit_out.eval_components()[comp.prefix])


	def choose_model(self,m1,m2,n_it,sc,mask_wl=''):
		"""Select the model with lowest reduced chi2 value, choosing between model m1 and m2, fitted over n_it iterations and using a sigma-clip value sc in each iteration. The model with the lowest chi2 will be stored in final_pars and final_fits. The use of the functions trim_to_range and subtract_fit makes that choose_model should be used for lines only."""
		fit_out_initial = self.fit_out
		self.set_models_iter(model=m1)
		try:
			self.trim_to_range(line='yes')
			if len(mask_wl)!=0:
				self.mask(mask_wl)
		except SpectrumError:
			raise
			return
		self.subtract_fit()
		for ii in range(n_it):
			try:
				self.run_fit()
			except SpectrumError:
				raise
				return
			if ii != n_it-1:
				self.use_fit_results(sigma_clip=sc)
		fit_out_copy = self.fit_out; cmod_copy = self.cmod; data_fit_copy = self.data_fit
		self.fit_out = fit_out_initial; del self.cmod
		self.set_models_iter(model=m2)
		try:
			self.trim_to_range(line='yes')
			if len(mask_wl)!=0:
				self.mask(mask_wl)
		except SpectrumError:
			raise
			return
		self.subtract_fit()
		for ii in range(n_it):
			try:
				self.run_fit()
			except SpectrumError:
				raise
				return
			if ii != n_it-1:
				z_est = self.fit_out.params['MgIIcenter'].value/2798 -1
				self.use_fit_results(sigma_clip=sc,limits={'MgIIsigma':(10,40),'MgIIcenter':(2795*(1+z_est),2801*(1+z_est)),'MgIIwcenter':(2795*(1+z_est),2801*(1+z_est))})
		if fit_out_copy.aic < self.fit_out.aic or self.fit_out.params['MgIIamplitude']<0 or self.fit_out.params['MgIIwamplitude']<0:
			for name in self.cmod.param_names:
				del self.final_pars[name]
				del self.final_pars[name+'Err']
			for comp in self.cmod.components:
				del self.final_fits[comp.prefix]
			self.fit_out = fit_out_copy; del fit_out_copy
			self.cmod = cmod_copy; del cmod_copy
			self.data_fit = data_fit_copy; del data_fit_copy
			for name in self.cmod.param_names:
				self.final_pars[name] = self.fit_out.params[name].value
				self.final_pars[name+'Err'] = self.fit_out.params[name].stderr
			for comp in self.cmod.components:
				self.final_fits[comp.prefix] = (self.data_fit[0],self.fit_out.eval_components()[comp.prefix])


	def subtract_fit(self):
		'''Subtract the result of the continuum (and possibly the FeII) fit(s) from the data, to fit only the line'''
		if len(self.data_fit) != 3:
			self.data_fit = self.data_fit.T
		if 'Fe'+self.line+'amp' in self.fit_out.params:
			self.data_fit[1] = self.data_fit[1] - powerLaw(self.data_fit[0],self.fit_out.params['pow'+self.line+'amp'].value,self.fit_out.params['pow'+self.line+'exp'].value)\
								-FeUV(self.data_fit[0],self.fit_out.params['FeUVredshift'].value,self.fit_out.params['FeUVamp'].value,self.fit_out.params['FeUVsigma'].value)
		else:
			self.data_fit[1] = self.data_fit[1] - powerLaw(self.data_fit[0],self.fit_out.params['pow'+self.line+'amp'],self.fit_out.params['pow'+self.line+'exp'])


	def output_text(self,dir_name='Alternative_MgII',name_add=''):
		"""Create a summary of the output in ascii and save to file"""
		savepathdata = self.path+'data/'+dir_name+'/'+self.obj+'_'+self.obs_date+name_add
		with open(savepathdata,'w') as f:
			for name in sorted(self.final_pars.iterkeys()):
				f.write(name.ljust(25)+str(self.final_pars[name])+'\n')
			f.write('chi2'.ljust(25)+str(self.fit_out.chisqr)+'\n')
			f.write('redchi2'.ljust(25)+str(self.fit_out.redchi)+'\n')


	def create_figure(self,dir_name='Alternative_MgII',fit='continuum',name_add = '',extra_spec='',fig_format='png'):
		"""Create a figure of the fit and save it"""
		savepathfigs = self.path+'figures/'+dir_name+'/'+self.obj+'_'+self.obs_date+'_'+fit+name_add+'.'+fig_format
		fig = plt.figure(1, figsize=(9,6))
		plt.minorticks_on()
		clr = {'pow'+self.line:('r','Continuum'),'FeUV':('y','FeII Template'),'MgII':('g','MgII Gaussian'),'MgIIw':('g','MgII Gaussian (B)'),'Hbn':('g',''),'Hbb':('g',''),'OIII_4959':('y',''),'OIII_5007':('y',''),'OIII_5007b':('y','')}
		if fit=='combined':
			plt.plot(self.data.T[0]/(1+self.z),self.data.T[1]*1e17,'b')
			tot_fit = np.add(self.final_fits['pow'+self.line][1],self.final_fits['FeUV'][1])
			for name,model in self.final_fits.iteritems():
				if name[:3] != 'pow' and name[:3]!='FeU':
					start = 0
					for ii in range(len(self.final_fits['FeUV'][0])):
						for jj in range(start,len(model[0])):
							if self.final_fits['FeUV'][0][ii] == model[0][jj]:
								tot_fit[ii] += model[1][jj]; start = jj
								break
				if name in clr:
					if name=='MgIIw':
						plt.plot(model[0]/(1+self.z),model[1]*1e17,c=clr[name][0],lw=1.5,label='_nolegend_')
					else:
						plt.plot(model[0]/(1+self.z),model[1]*1e17,c=clr[name][0],lw=1.5,label=clr[name][1])
				else:
					plt.plot(model[0]/(1+self.z),model[1]*1e17,c='k')
			plt.plot(self.final_fits['FeUV'][0]/(1+self.z),tot_fit*1e17,c='orange',lw=1.5)
			plt.ylim(0,np.amax(tot_fit)*1.2e17)
			plt.xlabel(r'Wavelength in QSO Restframe ($\rm \AA$)',fontsize=19, fontweight='bold')
			plt.ylabel(r'f$_\lambda$ (10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm\AA^{-1}$)',fontsize=19, fontweight='bold')
			plt.legend(loc='upper left', prop={'size': 16})
			plt.tick_params(axis='both', which='major', labelsize=17)
			plt.xlim(self.final_fits['FeUV'][0][0]/(1+self.z),3050)
			txtstr = '\n'.join((r'$f_{2798}$ = 15.4$\pm$0.06$\cdot$10$^{-16}$',r'$f_{MgII}$  = 9.02$\pm$0.58$\cdot$10$^{-15}$',r'$\sigma_{MgII}$ = 29.97$\pm$3.05 $\rm\AA$'))
			plt.text(0.6,0.75,txtstr,transform=fig.transFigure,fontsize=18,bbox=dict(boxstyle='round',facecolor='None',edgecolor='k'))
		elif fit=='combined_nofeii':
			plt.plot(self.data.T[0]/(1+self.z),self.data.T[1]*1e17,'b')
			tot_fit = np.copy(self.final_fits['pow'+self.line][1])
			for name,model in self.final_fits.iteritems():
				if name[:3] != 'pow':
					start = 0
					for ii in range(len(self.final_fits['pow'+self.line][0])):
						for jj in range(start,len(model[0])):
							if self.final_fits['pow'+self.line][0][ii] == model[0][jj]:
								tot_fit[ii] += model[1][jj]; start = jj
								break
				if name in clr:
					plt.plot(model[0]/(1+self.z),model[1]*1e17,c=clr[name])
				else:
					plt.plot(model[0]/(1+self.z),model[1]*1e17,c='k')
			plt.plot(self.final_fits['pow'+self.line][0]/(1+self.z),tot_fit*1e17,c='orange',lw=1)
			plt.xlim(self.lower/(1+self.z),self.upper/(1+self.z))
			plt.ylim(0,np.amax(tot_fit)*1.2e17)
			plt.xlabel(r'Wavelength in QSO Restframe ($\rm \AA$)',fontsize=14, fontweight='bold')
			plt.ylabel(r'f$_\lambda$ (10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm\AA^{-1}$)',fontsize=14, fontweight='bold')			
		else:
			plt.plot(self.data_fit[0], self.data_fit[1], 'b')
			plt.plot(self.data_fit[0], self.fit_out.best_fit, 'r')
			for jj in self.cmod.components:
				if jj.prefix == 'pow'+self.line:
					plt.plot(self.data_fit[0], self.fit_out.eval_components()[jj.prefix], linestyle='dashed', color='g')
				elif jj.prefix == 'FeUV':
					plt.plot(self.data_fit[0], self.fit_out.eval_components()[jj.prefix], color='c')	
				else:
					plt.plot(self.data_fit[0], self.fit_out.eval_components()[jj.prefix], color='g')
		plt.ylim(bottom=0)
		plt.tight_layout()
		plt.savefig(savepathfigs)
		plt.clf()

