import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit, fsolve, leastsq
import scipy.stats as stat
import scipy.integrate as integrate
from statsmodels.stats.diagnostic import lilliefors
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcl
import matplotlib.patches as mpt
from astropy.time import Time
import astropy.units as unit
from astropy.cosmology import Planck15 as cosmo


class Normalisation:
	"""Contains functions to normalise the input data (line and continuum fluxes) to the the epoch of highest continuum state"""
	
	def __init__(self,namelist=[],spec_namelist=[],exclude=[],fits_name=None,SN_limit=None,deltat_limit=None):
		if len(exclude)!=0: exclude = np.loadtxt(exclude,dtype='str')
		try:
			fin = np.loadtxt(namelist,dtype={'names':('name','obj','ra','dec','z','dt','airmass','S/N','A_v_sf','A_v_sdss'),'formats':('S30','S30','float','float','float','float','float','float','float','float')})
			self.namelist,self.sdss_namelist,self.zdt_list = [],[],[]
			#This line is to account for an idiosyncrasy in the data format, where the same object is listed under different names
			self.simlist = []
			ii=0
			while ii < len(fin['obj']):
				if fin['name'][ii] in exclude:
					if ii%2==1:
						del self.namelist[-1]
						del self.sdss_namelist[-1]
						del self.zdt_list[-1]
						ii+=1
					else:
						ii+=2
					continue
				if deltat_limit:
					if fin['dt'][ii]<deltat_limit:
						ii+=2
						continue
					else:
						if SN_limit:
							if fin['S/N'][ii]<SN_limit:
								if ii%2==1:
									del self.namelist[-1]
									del self.sdss_namelist[-1]
									del self.zdt_list[-1]
									ii+=1
								else:
									ii+=2
								continue
							else:
								if ii%2==1 and fin['obj'][ii]!=self.namelist[-1][:len(fin['obj'][ii])]:
									if fin['z'][ii]==self.zdt_list[-1][0] and fin['dt'][ii]==self.zdt_list[-1][1]:
										self.simlist += [fin['obj'][ii][:9]]
								self.zdt_list += [[ fin['z'][ii],fin['dt'][ii],fin['S/N'][ii] ]]
								self.sdss_namelist += [fin['name'][ii]]
								if fin['name'][0]=='/':
									self.namelist += [fin['obj'][ii]+'_'+fin['name'][ii][12:17]]
								else:
									self.namelist += [fin['obj'][ii]+'_'+fin['name'][ii][10:15]]
							ii+=1
							continue
				if SN_limit:
					if fin['S/N'][ii]<SN_limit:
						if ii%2==1:
							del self.namelist[-1]
							del self.sdss_namelist[-1]
							del self.zdt_list[-1]
							ii+=1
						else:
							ii+=2
						continue
					else:
						if ii%2==1 and fin['obj'][ii]!=self.namelist[-1][:len(fin['obj'][ii])]:
							if fin['z'][ii]==self.zdt_list[-1][0] and fin['dt'][ii]==self.zdt_list[-1][1]:
								self.simlist += [fin['obj'][ii][:9]]
						self.zdt_list += [[ fin['z'][ii],fin['dt'][ii],fin['S/N'][ii] ]]
						self.sdss_namelist += [fin['name'][ii]]
						if fin['name'][0]=='/':
							self.namelist += [fin['obj'][ii]+'_'+fin['name'][ii][12:17]]
						else:
							self.namelist += [fin['obj'][ii]+'_'+fin['name'][ii][10:15]]
					ii+=1
					continue
				#Inclusion in case no delta_t limit or S/N limit is set
				if ii%2==1 and fin['obj'][ii]!=self.namelist[-1][:len(fin['obj'][ii])]:
					if fin['z'][ii]==self.zdt_list[-1][0] and fin['dt'][ii]==self.zdt_list[-1][1]:
						self.simlist += [fin['obj'][ii][:9]]
				self.zdt_list += [[ fin['z'][ii],fin['dt'][ii],fin['S/N'][ii] ]]
				self.sdss_namelist += [fin['name'][ii]]
				if fin['name'][0]=='/':
					self.namelist += [fin['obj'][ii]+'_'+fin['name'][ii][12:17]]
				else:
					self.namelist += [fin['obj'][ii]+'_'+fin['name'][ii][10:15]]
				ii+=1

			self.namelist = np.array(self.namelist)
			self.sdss_namelist = np.array(self.sdss_namelist)
			self.zdt_list = np.array(self.zdt_list)
		except:
			print 'Loading filename only list'
			self.namelist = np.loadtxt(namelist,dtype='str')
			self.simlist = []; self.zdt_list = len(self.namelist)*[[0,0,0]]; self.sdss_namelist=np.loadtxt(spec_namelist,dtype='str')
			for ii in range(len(self.namelist)):
				if self.namelist[ii] in exclude: np.delete(self.namelist,ii)

		self.fits_name = fits_name
		self.normalised = 0
		self.data = None

	@staticmethod
	def date_converter(array_in):
		'''search through a given array of dates and convert non MJD-dates to MJD. Assumed format for non-MJD dates is YYYYMMDD'''
		for ii in range(len(array_in)):
			if len(str(array_in[ii]))!=5:
				d = str(array_in[ii])
				d = d[:4]+'-'+d[4:6]+'-'+d[6:]
				t = Time(d,format='iso')
				array_in[ii] = int(t.mjd)
		return np.array(array_in)


	def load_data(self,data_path='./',conf_columns=[21,22,29,30],linef_columns=[19,20,27,28],distance_col = 31,inc_double_sig=False):
		'''load and organize the data appropriately. Order for columns: flux1, flux1_err, flux2, flux2_err. This function is very specific to the data format.'''

		cc,lc,dc = conf_columns,linef_columns,distance_col

		Line, Line_err = [],[]
		C, C_err = [],[]
		SDSS_List, Name_List, Date_List, Z_List,Sigma_List,Sigma_err = [],[],[],[],[],[]
		Z_Est, DT, SN = [],[],[]

		#Load data from lmfit results, over multiple objects
		if len(self.namelist)!=0:

			line_, line_err = [],[]
			c,c_err = [],[]
			sdss_list,name_list,date_list,z_list,sigma_list,sigma_err = [],[],[],[],[],[]

			namelen = self.namelist[0].find('_')
			name = self.namelist[0][:namelen]; date = self.namelist[0][namelen+1:]
			self.nobj = 0
			for ii in range(len(self.namelist)):

				spec = self.namelist[ii]
				namelen = spec.find('_')
				date = spec[namelen+1:]

				if spec[:namelen]!=name: #Group together spectra for the same object
					if spec[:9] in self.simlist and len(name_list)==1: #Account for two names being used for the same object
						pass
					else:
						name = spec[:namelen]

						Line+=[line_]; Line_err+=[line_err]
						C+=[c]; C_err+=[c_err]
						Name_List+=[name_list]
						SDSS_List+=[sdss_list]
						Date_List += [date_list]
						Z_List += [z_list]
						Z_Est += [len(z_list)*[self.zdt_list[ii-1][0]]]
						Sigma_List += [sigma_list]
						Sigma_err+=[sigma_err]
						if self.zdt_list[ii-1][2]!=0:
							DT += [len(z_list)*[self.zdt_list[ii-1][1]]]
							SN += [[self.zdt_list[ii-1][2],self.zdt_list[ii][2]]]
						else:
							date_list = self.date_converter(date_list)
							dt = []
							for jj in range(len(date_list)-1):
								dt += [abs(int(date_list[jj])-int(date_list[jj+1]))]
							dt+=[dt[-1]]
							DT += [dt]
							SN += [len(z_list)*[0]]
						line_, line_err = [],[]
						c, c_err = [],[]
						sdss_list,name_list,date_list,z_list,sigma_list,sigma_err = [],[],[],[],[],[]
						self.nobj+=1

				#Load fitting results
				if spec.find('+')!=-1: spec_load = spec.replace('+','_')
				else: spec_load = spec.replace('-','_')
				fi = np.loadtxt(data_path+spec_load,dtype={'names':('par','val'),'formats':('S20','S30')})

				ampb,ampb_e = 0,0
				sigw=0
				for row in fi:
					if row[1]=='None':
						val=0
					else:
						val = float(row[1])
					if row[0]=='powUVamp': amp=val
					elif row[0]=='powUVampErr': ampErr=val
					elif row[0]=='powUVexp': exp=val
					elif row[0]=='powUVexpErr': expErr=val
					elif row[0]=='MgIIamplitude': ampb+=val
					elif row[0]=='MgIIamplitudeErr': ampb_e+=val*val
					elif row[0]=='MgIIcenter': centb=val
					elif row[0]=='MgIIsigma': sigb=val
					elif row[0]=='MgIIsigmaErr': sigb_e=val
					elif row[0]=='MgIIwamplitude': ampb+=val
					elif row[0]=='MgIIwamplitudeErr': ampb_e+=val*val
					elif row[0]=='MgIIwsigma': sigw=val
					elif row[0]=='MgIIwsigmaErr': sigw_e=val
				c+= [amp*np.power(centb/5100.,exp)]
				c_err+= [np.power((centb/5100.0),exp)*np.sqrt(np.power(ampErr,2) + np.power((amp*np.log(centb/5100.0)*expErr),2))]
				line_+= [ampb]
				line_err += [np.sqrt(ampb_e)]
				name_list += [name]
				sdss_list += [self.sdss_namelist[ii]]
				date_list += [date]
				if sigw!=0:
					if inc_double_sig==True:
#						print name, date
						sigma_list += [np.maximum(sigb,sigw)]
						sigma_err += [ (sigb_e,sigw_e)[np.argmax((sigb,sigw))] ]
					else: sigma_list+=[0]; sigma_err+=[0]
				else:
					sigma_list += [sigb]; sigma_err+=[sigb_e]
				z_list += [centb/2798. - 1.]

				if spec==self.namelist[-1]:#Include last spectrum in the list
					Line+=[line_]; Line_err+=[line_err]
					C+=[c]; C_err+=[c_err]
					Name_List+=[name_list]
					SDSS_List+=[sdss_list]
					Date_List += [date_list]
					Z_List += [z_list]
					Z_Est += [len(z_list)*[self.zdt_list[ii][0]]]
					if self.zdt_list[ii-1][2]!=0:
						DT += [len(z_list)*[self.zdt_list[ii-1][1]]]
						SN += [[self.zdt_list[ii-1][2],self.zdt_list[ii][2]]]
					else:
						date_list = self.date_converter(date_list)
						dt = []
						for jj in range(len(date_list)-1):
							dt += [abs(int(date_list[jj])-int(date_list[jj+1]))]
						dt+=[dt[-1]]
						DT += [dt]
						SN += [len(z_list)*[0]]
					Sigma_List+=[sigma_list]
					Sigma_err+=[sigma_err]
					self.nobj+=1
					
			self.data = np.array([SDSS_List,Name_List,Date_List,C,C_err,Line,Line_err,Z_List,Z_Est,DT,SN,Sigma_List,Sigma_err])


		#Load data from the fits file (QSFit data)
		elif self.fits_name!=None:
			hdulist = fits.open(self.fits_name)
			tb = hdulist[1].data

			for row in tb:
				if row[lc[0]]!=0:
					C += [[row[cc[0]],row[cc[2]]]]
					C_err += [[row[cc[1]],row[cc[3]]]]
					Line += [[row[lc[0]],row[lc[2]]]]
					Line_err += [[row[lc[1]],row[lc[3]]]]
					Name_List += [[row[0],row[0]]]
					if val[5:8]=='wht' or val[5:8]=='mmt':
						Date_List += [[0,int(val[9:17])]]
					elif val[5:8]=='mag':
						Date_List += [[0,int('20'+val[16:22])]]

			self.data = [Name_List,Date_List,C,C_err,Line,Line_err]

		else: print 'No data file specified'; return


	def load_mc_errors(self,data_path='\.',nmc=100,inc_double_sig=False,ew=False):
		'''Replace the errors in the data attribute with an average of the errors from a Monte Carlo simulation (N=nmc)'''

		C_err,Line_err,Sigma_err,EW_err = [],[],[],[]

		#Load data from lmfit results, over multiple objects
		if len(self.namelist)!=0:

			c_err,line_err,sigma_err,ew_err = [],[],[],[]

			namelen = self.namelist[0].find('_')
			name = self.namelist[0][:namelen]; date = self.namelist[0][namelen+1:]
			for ii in range(len(self.namelist)):
				spec = self.namelist[ii]
				namelen = spec.find('_')

				if spec[:namelen]!=name: #Group together spectra for the same object
					if spec[:9] in self.simlist and len(name_list)==1: #Account for two names being used for the same object
						pass
					else:
						name = spec[:namelen]

						Line_err+=[line_err]
						C_err+=[c_err]
						Sigma_err += [sigma_err]

						c_err,line_err,sigma_err = [],[],[]

				#Load fitting results
				if spec.find('+')!=-1: spec_load = spec.replace('+','_')
				else: spec_load = spec.replace('-','_')
				c_mc,l_mc,sig_mc,ew_mc=[],[],[],[]
				for mm in range(nmc):
					try:
						fi = np.loadtxt(data_path+spec_load+'_'+str(mm),dtype={'names':('par','val'),'formats':('S20','S30')})
					except IOError: continue

					ampb,ampb_e = 0,0
					sigw=0; G2=0
					for row in fi:
						if row[1]=='None':
							val=0
						else:
							val = float(row[1])
						if row[0]=='powUVamp': amp=val
						elif row[0]=='powUVexp': exp=val
						elif row[0]=='MgIIamplitude': ampb+=val
						elif row[0]=='MgIIcenter': cb=val
						elif row[0]=='MgIIsigma': sb=val
						elif row[0]=='MgIIwamplitude': G2=1; ampw=val; ampb+=val
						elif row[0]=='MgIIwcenter': cw=val
						elif row[0]=='MgIIwsigma': sw=val
					c_mc+= [amp*np.power(cb/5100.,exp)]
					l_mc+= [ampb]
					if sigw!=0:
						if inc_double_sig==True:
							sig_mc += [np.average(sb,sw)]
						else: sig_mc+=[0]
					else:
						sig_mc += [sb]
					if ew==True:
						if G2 != 0:
							I = integrate.quad(self.p_EqW2,cb-8*sb,cb+8*sb,args=(ampb,cb,sb,ampw,cw,sw,amp,exp))
							ew_mc+=[I[0]]
						else:
							I = integrate.quad(self.p_EqW1,cb-8*sb,cb+8*sb,args=(ampb,cb,sb,amp,exp))
							ew_mc+=[I[0]]
					else:
						ew_mc+=[0]
				c_err+=[np.std(c_mc)]; line_err+=[np.std(l_mc)]; sigma_err+=[np.std(sig_mc)]; ew_err+=[np.std(ew_mc)]

				if spec==self.namelist[-1]:#Include last spectrum in the list
					C_err+=[c_err]; Line_err+=[line_err]; Sigma_err+=[sigma_err]; EW_err+=[ew_err]

			#Replace the errors in the data attribute
			if ew==True:
				self.data[4]=C_err; self.data[6]=Line_err; self.data[8]=EW_err
			else:
				self.data[4]=C_err; self.data[6]=Line_err; self.data[12]=Sigma_err


	def Gaussian(self,x,amp,mu,sigma):
		'''needed in the EW calculation'''
		return amp/(sigma*np.sqrt(2*np.pi))*np.exp(-.5*np.power((x-mu)/sigma,2))

	def PL(self,x,amp,ind):
		'''needed in the EW calculation'''
		return amp*np.power(x/5100.,ind)

	def p_EqW1(self,x,Ga,Gm,Gs,PLa,PLi):
		'''needed in the EW calculation'''
		return self.Gaussian(x,Ga,Gm,Gs)/self.PL(x,PLa,PLi)

	def p_EqW2(self,x,G1a,G1m,G1s,G2a,G2m,G2s,PLa,PLi):
		'''needed in the EW calculation'''
		return (self.Gaussian(x,G1a,G1m,G1s)+self.Gaussian(x,G2a,G2m,G2s))/self.PL(x,PLa,PLi)

	def load_data_ew(self,data_path='./'):
		'''Load data from the spectral fit output files, and calculate the equivalent widths and associated errors'''
		Line, Line_err = [],[]
		C, C_err = [],[]
		EW, EW_err = [],[]
		SDSS_List, Name_List, Date_List, Z_List = [],[],[],[]

		#Load data from lmfit results, over multiple objects
		if len(self.namelist)!=0:

			line_, line_err = [],[]
			c, c_err = [],[]
			ew, ew_err = [],[]
			sdss_list,name_list,date_list,z_list  = [],[],[],[]

			namelen = self.namelist[0].find('_')
			name = self.namelist[0][:namelen]; date = self.namelist[0][namelen+1:]
			self.nobj = 0
			for ii in range(len(self.namelist)):

				spec = self.namelist[ii]
				namelen = spec.find('_')
				date = spec[namelen+1:]

				if spec[:namelen]!=name: #Group together spectra for the same object
					if spec[:9] in self.simlist and len(name_list)==1: #Account for two names being used for the same object
						pass
					else:
						name = spec[:namelen]

						Line+=[line_]; Line_err+=[line_err]
						C+=[c]; C_err+=[c_err]
						EW+=[ew]; EW_err+=[ew_err]
						Name_List+=[name_list]
						SDSS_List+=[sdss_list]
						Date_List += [date_list]
						Z_List += [z_list]

						line_, line_err = [],[]
						c, c_err = [],[]
						ew, ew_err = [],[]
						sdss_list,name_list,date_list,z_list = [],[],[],[]
						self.nobj+=1

				#Load fitting results
				if spec.find('+')!=-1: spec_load = spec.replace('+','_')
				else: spec_load = spec.replace('-','_')
				fi = np.loadtxt(data_path+spec_load,dtype={'names':('par','val'),'formats':('S20','S30')})

				lf, lfe = 0,0
				G2 = 0
				for row in fi:
					if row[1]=='None':
						val=0
					else:
						val = float(row[1])
					if row[0]=='powUVamp': amp=val
					elif row[0]=='powUVampErr': ampErr=val
					elif row[0]=='powUVexp': exp=val
					elif row[0]=='powUVexpErr': expErr=val
					elif row[0]=='MgIIamplitude': 
						lf+=val; ampb=val
					elif row[0]=='MgIIamplitudeErr':
						lfe+=np.power(val,2); ampb_err=val
					elif row[0]=='MgIIcenter': cb=val
					elif row[0]=='MgIIsigma': sb=val
					elif row[0]=='MgIIwamplitude':
						G2 = 1 #Marker that a second Gaussian was used in the fitting
						lf+=val; ampw=val
					elif row[0]=='MgIIwamplitudeErr':
						lfe+=np.power(val,2); ampw_err=val
					elif row[0]=='MgIIwcenter': cw=val
					elif row[0]=='MgIIwsigma': sw=val
				c+= [amp*np.power(cb*(3000./2798.)/5100.0,exp)]
				c_err+= [np.power((cb*(3000./2798.)/5100.0),exp)*np.sqrt(np.power(ampErr,2) + np.power((amp*np.log(cb*(3000./2798.)/5100.0)*expErr),2))] 
				line_+= [lf]
				line_err+= [np.sqrt(lfe)]
				pl_tmp = self.PL(cb,amp,exp)
				pl_err_tmp = np.power((cb/5100.0),exp)*np.sqrt(np.power(amp,2) + np.power((amp*np.log(cb/5100.0)*exp),2))
				if G2 != 0:
					I = integrate.quad(self.p_EqW2,cb-8*sb,cb+8*sb,args=(ampb,cb,sb,ampw,cw,sw,amp,exp))
					ew+=[I[0]]
					ampb_err = np.sqrt(ampb_err*ampb_err + ampw_err+ampw_err)
				else:
					I = integrate.quad(self.p_EqW1,cb-8*sb,cb+8*sb,args=(ampb,cb,sb,amp,exp))
					ew+=[I[0]]
				ew_err += [np.sqrt( np.power(ampb_err/pl_tmp,2) + np.power(ampb_err*pl_err_tmp,2)/np.power(pl_tmp,4) )]
				name_list += [name]
				sdss_list += [self.sdss_namelist[ii]]
				date_list += [date]
				z_list += [cb/2798. - 1.]

				if spec==self.namelist[-1]:#Include last spectrum in the list
					Line+=[line_]; Line_err+=[line_err]
					C+=[c]; C_err+=[c_err]
					EW += [ew]; EW_err+=[ew_err]
					Name_List+=[name_list]
					SDSS_List+=[sdss_list]
					Date_List += [date_list]
					Z_List += [z_list]
					self.nobj+=1

			self.data = np.array([SDSS_List,Name_List,Date_List,C,C_err,Line,Line_err,EW,EW_err,Z_List])

			
	def add_sdss_dates(self,datefile='sdssnames.txt'):
		'''Find the dates for sdss spectra (to be used in combination with data from fits file)'''
		indata = np.loadtxt(datefile,dtype={'names':('a','name','c','spec'),'formats':(int,'S7',float,'S25')})
		for ii in range(len(self.data[0])):
			name = self.data[0][ii,0]
			if name in indata['name']:
				self.data[1][ii,0] = int(indata['spec'][np.argwhere(indata['name']==name)][0][0][10:15])

	def normalise(self,n='cmax'):
		'''Normalise the flux data, as descibed in the paper. Setting n='cmax' normalises to the date of maximum continuum, and n='cmin' to the date of minimum continuum'''
		DS_List, DC_List,DL_List = [],[],[]
		DSe_List, DCe_List,DLe_List = [],[],[]
		for ii in range(len(self.data[0])):
			cpc,cpl,cps = np.array(self.data[3][ii]).astype('float'),np.array(self.data[5][ii]).astype('float'),np.array(self.data[11][ii]).astype('float')
			cpce,cple,cpse = np.array(self.data[4][ii]).astype('float'),np.array(self.data[6][ii]).astype('float'),np.array(self.data[12][ii]).astype('float')
			if n=='cmax':
				nc = np.amax(cpc); nl = cpl[np.argmax(cpc)]
			elif n=='cmin':
				nc = np.amin(cpc); nl= cpl[np.argmin(cpc)]
			DS,DC,DL = [],[],[]
			DSe,DCe,DLe = [],[],[]
			for jj in range(len(cpc)-1):
				DC += [(float(cpc[jj])-float(cpc[jj+1]))/cpc[jj]]
				DCe+= [np.sqrt(np.power(cpce[jj+1]/cpc[jj],2)+np.power(cpc[jj+1]*cpce[jj]/(cpc[jj]*cpc[jj]),2))]
				DL += [(float(cpl[jj])-float(cpl[jj+1]))/cpl[jj]]
				DLe+= [np.sqrt(np.power(cple[jj+1]/cpl[jj],2)+np.power(cpl[jj+1]*cple[jj]/(cpl[jj]*cpl[jj]),2))]
				if cps[jj]==0 or cps[jj+1]==0: DS+=[0]; DSe+=[0]
				else: DS += [(cps[jj]-cps[jj+1])/cps[jj]]; DSe+=[np.sqrt(np.power(cpse[jj+1]/cps[jj],2)+np.power(cps[jj+1]*cpse[jj]/(cps[jj]*cps[jj]),2))] #Normalise del_sigma to oldest epoch, assuming all epochs are in chr. order
			DS += [DS[-1]]; DC += [DC[-1]]; DL+=[DL[-1]]
			DSe += [DSe[-1]]; DCe += [DCe[-1]]; DLe+=[DLe[-1]]
			DS_List += [DS]; DC_List += [DC]; DL_List += [DL]
			DSe_List += [DSe]; DCe_List += [DCe]; DLe_List += [DLe]
			self.data[3][ii] = cpc/nc; self.data[5][ii] = cpl/nl
			self.data[4][ii] = np.array(self.data[4][ii]).astype('float')/nc; self.data[6][ii] = np.array(self.data[6][ii]).astype('float')/nl
		DC_List = np.array([DS_List,DSe_List,DC_List,DCe_List,DL_List,DLe_List])
		self.data = np.concatenate((self.data,DC_List),axis=0)
		self.normalised=True

	def create_flux_list(self,name_add=''):
		'''Write the normalised fluxes and previously data into a single file \n
		format of input data: np.array([SDSS_List 0,Name_List 1,Date_List 2,C 3,C_err 4,Line 5,Line_err 6,Z_List 7,Z_Est 8,DT 9,SN 10,Sigma_List 11, Sigma_err 12])'''
		if self.normalised:
			outfile='Named_MgII_norm_flux'+name_add
			with open(outfile,'w') as of:
				of.write( '#{:>30} {:>35} {:>23} {:>25} {:>23} {:>24} {:>20} {:>20} {:>11} {:>15} {:>18} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}\n'.format('SpecName','Name','Continuum','Continuum_err','Line','Line_err','z_fit','z_sdss','Delta_t','S/N_sdss','Sigma','Sigma_err','Delta_S','Delta_S_err','Delta_C','Delta_C_err','Delta_L','Delta_L_err') )
				for ii in range(len(self.data[0])):
					for jj in range(len(self.data[0][ii])):
						of.write( ' {:>30} {:>35} {:>23} {:>25} {:>28} {:>24} {:>20} {:>20} {:>11} {:>15} {:>18} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}\n'.format( self.data[0][ii][jj],self.data[1][ii][jj]+'_'+str(self.data[2][ii][jj]),self.data[3][ii][jj],self.data[4][ii][jj],self.data[5][ii][jj],self.data[6][ii][jj],self.data[7][ii][jj],self.data[8][ii][jj],self.data[9][ii][jj],str(round(float(self.data[10][ii][jj]),2)),self.data[11][ii][jj],self.data[12][ii][jj],self.data[13][ii][jj],self.data[14][ii][jj],self.data[15][ii][jj],self.data[16][ii][jj],self.data[17][ii][jj],self.data[18][ii][jj] ) )
			outfile='MgII_norm_flux'+name_add
			with open(outfile,'w') as of:
				of.write( '#{:>23} {:>25} {:>23} {:>24} {:>20} {:>20} {:>16} {:>15} {:>18} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}\n'.format('Continuum','Continuum_err','Line','Line_err','z_fit','z_sdss','Delta_t','S/N_sdss','Sigma','Sigma_err','Delta_S','Delta_S_err','Delta_C','Delta_C_err','Delta_L','Delta_L_err') )
				for ii in range(len(self.data[0])):
					for jj in range(len(self.data[0][ii])):
						of.write( ' {:>23} {:>25} {:>23} {:>24} {:>20} {:>20} {:>16} {:>15} {:>18} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}\n'.format( str(self.data[3][ii][jj]),str(self.data[4][ii][jj]),str(self.data[5][ii][jj]),str(self.data[6][ii][jj]),str(self.data[7][ii][jj]),str(self.data[8][ii][jj]),str(self.data[9][ii][jj]),str(self.data[10][ii][jj]),str(self.data[11][ii][jj]),str(self.data[12][ii][jj]),self.data[13][ii][jj],self.data[14][ii][jj],self.data[15][ii][jj],self.data[16][ii][jj],self.data[17][ii][jj],self.data[18][ii][jj] ) )
		else:
			DC_List,DCe_List,DL_List,DLe_List,DS_List,DSe_List = [],[],[],[],[],[]
			for ii in range(len(self.data[0])):
				cpc,cpce,cpl,cple = np.array(self.data[3][ii]).astype('float'),np.array(self.data[4][ii]).astype('float'),np.array(self.data[5][ii]).astype('float'),np.array(self.data[6][ii]).astype('float')
				cps,cpse = np.array(self.data[11][ii]).astype('float'),np.array(self.data[12][ii]).astype('float')
				DC,DCe,DL,DLe,DS,DSe=[],[],[],[],[],[]
				for jj in range(len(cpc)-1):
					DC += [float(cpc[jj])-float(cpc[jj+1])]; DCe += [np.sqrt(cpce[jj]*cpce[jj]+cpce[jj+1]*cpce[jj+1])]
					DL += [float(cpl[jj])-float(cpl[jj+1])]; DLe += [np.sqrt(cple[jj]*cple[jj]+cple[jj+1]*cple[jj+1])]
					if cps[jj]==0 or cps[jj+1]==0: DS+=[0]; DSe+=[0]
					else: DS += [cps[jj]-cps[jj+1]]; DSe+=[np.sqrt(cpse[jj]*cpse[jj]+cpse[jj+1]*cpse[jj+1])]
				DC+=[DC[-1]]; DCe+=[DCe[-1]]; DL+=[DL[-1]]; DLe+=[DLe[-1]]; DS+=[DS[-1]]; DSe+=[DSe[-1]]
				DC_List += [DC]; DCe_List+=[DCe]; DL_List += [DL]; DLe_List+=[DLe]; DS_List+=[DS]; DSe_List+=[DSe]
			DC_List = np.array([DS_List,DSe_List,DC_List,DCe_List,DL_List,DLe_List])
			self.data = np.concatenate((self.data,DC_List),axis=0)
			outfile='Named_MgII_flux'+name_add
			with open(outfile,'w') as of:
				of.write( '#{:>30} {:>18} {:>25} {:>23} {:>20} {:>24} {:>16} {:>20} {:>10} {:>15} {:>25} {:>23} {:>25} {:>23} {:>25} {:>23} {:>25} {:>23}\n'.format('SpecName','Name','Continuum','Continuum_err','Line','Line_err','z_fit','z_sdss','Delta_t','S/N_sdss','Sigma','Sigma_err','Delta_S','Delta_S_err','Delta_C','Delta_C_err','Delta_L','Delta_L_err') )
				for ii in range(len(self.data[0])):
					for jj in range(len(self.data[0][ii])):
						of.write( ' {:>30} {:>18} {:>25} {:>23} {:>20} {:>24} {:>16} {:>20} {:>10} {:>15} {:>25} {:>23} {:>25} {:>23} {:>25} {:>23} {:>25} {:>23}\n'.format( self.data[0][ii][jj],self.data[1][ii][jj]+'_'+str(self.data[2][ii][jj]),str(self.data[3][ii][jj]),str(self.data[4][ii][jj]),str(self.data[5][ii][jj]),str(self.data[6][ii][jj]),str(self.data[7][ii][jj]),str(self.data[8][ii][jj]),str(self.data[9][ii][jj]),str(self.data[10][ii][jj]),str(self.data[11][ii][jj]),str(self.data[12][ii][jj]),str(self.data[13][ii][jj]),str(self.data[14][ii][jj]),str(self.data[15][ii][jj]),str(self.data[16][ii][jj]),str(self.data[17][ii][jj]),str(self.data[18][ii][jj]) ) )
			outfile='MgII_flux'+name_add
			with open(outfile,'w') as of:
				of.write( '#{:>25} {:>23} {:>20} {:>24} {:>16} {:>20} {:>10} {:>15} {:>25} {:>23} {:>25} {:>23} {:>25} {:>23} {:>25} {:>23}\n'.format('Continuum','Continuum_err','Line','Line_err','z_fit','z_sdss','Delta_t','S/N_sdss','Sigma','Sigma_err','Delta_S','Delta_S_err','Delta_C','Delta_C_err','Delta_L','Delta_L_err') )
				for ii in range(len(self.data[0])):
					for jj in range(len(self.data[0][ii])):
						of.write( ' {:>25} {:>23} {:>20} {:>24} {:>16} {:>20} {:>10} {:>15} {:>25} {:>23} {:>25} {:>23} {:>25} {:>23} {:>25} {:>23}\n'.format( str(self.data[3][ii][jj]),str(self.data[4][ii][jj]),str(self.data[5][ii][jj]),str(self.data[6][ii][jj]),str(self.data[7][ii][jj]),str(self.data[8][ii][jj]),str(self.data[9][ii][jj]),str(self.data[10][ii][jj]),str(self.data[11][ii][jj]),str(self.data[12][ii][jj]),str(self.data[13][ii][jj]),str(self.data[14][ii][jj]),str(self.data[15][ii][jj]),str(self.data[16][ii][jj]),str(self.data[17][ii][jj]),str(self.data[18][ii][jj]) ) )

	def create_ew_list(self,name_add=''):
		'''Write the equivalent widths into a single file'''
		outfile='MgII_EW'+name_add
		with open(outfile,'w') as of:
			of.write( '#{:>25} {:>23} {:>20} {:>24} {:>25} {:>23} {:>23}\n'.format('Continuum','Continuum_err','Line','Line_err','EW','EW_err','z_est') )
			for ii in range(len(self.data[0])):
				for jj in range(len(self.data[0][ii])):
					of.write( ' {:>25} {:>23} {:>20} {:>24} {:>25} {:>23} {:>23}\n'.format(str(self.data[3][ii][jj]),str(self.data[4][ii][jj]),str(self.data[5][ii][jj]),str(self.data[6][ii][jj]),str(self.data[7][ii][jj]),str(self.data[8][ii][jj]),str(self.data[9][ii][jj]) ))
		outfile='Named_MgII_EW'+name_add
		with open(outfile,'w') as of:
			of.write( '#{:>30} {:>25} {:>23} {:>20} {:>24} {:>25} {:>23} {:>23}\n'.format('Name','Continuum','Continuum_err','Line','Line_err','EW','EW_err','z_est') )
			for ii in range(len(self.data[0])):
				for jj in range(len(self.data[0][ii])):
					of.write( ' {:>30} {:>25} {:>23} {:>20} {:>24} {:>25} {:>23} {:>23}\n'.format(self.data[1][ii][jj],str(self.data[3][ii][jj]),str(self.data[4][ii][jj]),str(self.data[5][ii][jj]),str(self.data[6][ii][jj]),str(self.data[7][ii][jj]),str(self.data[8][ii][jj]),str(self.data[9][ii][jj]) ))
					


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


class Fit:
	"""Contains models to be used in fitting, as well as the chi2 calculation, a method to produce errors on the fits, and methods to create figures"""

	def __init__(self,fname='fit_flat_inv'):
		self.fname=fname
		self.fit_func = None
		self.data = None
		self.namelist=''
		self.func= None
		self.npars = None
		self.par = None
		self.pcov = None
		self.Nspec = None
		self.residuals = None
		self.cor_coeff = None
		self.chi2 = None
		self.red_chi2 = None

	##################################################################################	
	##Define possible functions to be used in fits
	def fit_line(x,*p):
		a = p	
		return np.array(a*x)

	def fit_line_offset(x,*p):
		a,b = p
		return np.array(b+a*x)
	
	def fit_pl4(x,*p):
		x_k,a,b,c = p
		xt = x/x_k
		return np.power(xt,b)/(1+np.power(xt,a)) + np.power(xt,a+c)/(1+np.power(xt,a))

	def fit_pl3(x,*p):
		x_k,b,s = p
		xt = x/x_k
		return (xt+np.power(xt,b+s))/(1+np.power(xt,s))

	def fit_pl2(x,*p):
		x_k,b = p
		xt = x/x_k
		s = 4.
		return (xt+np.power(xt,b+s))/(1+np.power(xt,s))

	def fit_lin3(x,*p):
		if len(p) != 1:
			x_k,a,b = p
		else:
			x_k,a,b = p[0]
		if type(x) == np.ndarray:
			f = []
			for el in x:
				if el<x_k:
					f+=[el*a]
				else:
					f+=[a*x_k+(el-x_k)*b]
		else:
			if x<x_k:
				f=[x*a]
			else:
				f=[a*x_k+(x-x_k)*b]
		return np.array(f)

	def fit_lin2(x,*p):
		if len(p) != 1: #This statement is required to properly deal with a difference in data format for the parameters when passed through different functions.
			x_k,b=p
		else:
			x_k,b = p[0]
		if type(x) == np.ndarray:
			f = []
			for el in x:
				if el<x_k:
					f+=[np.float(el)]
				else:
					f+=[np.float(x_k+(el-x_k)*b)]
		else:
			if x<x_k:
				f=[np.float(x)]
			else:
				f=[np.float(x_k+(x-x_k)*b)]
		return np.array(f)

	def fit_pl_slopes(x,*p):
		x_k, a, b, s = p
		xt = x/x_k
		return (np.power(xt,a)+np.power(xt,b+s))/(1+np.power(x,s))

	def fit_flat(x,*p):
		if len(p)!=1:
			x_k,a = p
		else:
			x_k,a = p[0]
		if type(x) == np.ndarray:
			f = []
			for el in x:
				if el<x_k:
					f+=[np.float(el*a)]
				else:
					f+=[np.float(x_k*a)]
		else:
			if x<x_k:
				f = [np.float(x*a)]
			else:
				f = [np.float(x_k*a)]
		return np.array(f)

	def fit_flat_inv(x,*p):
		if len(p)!=1:
			y_s,a = p
		else:
			y_s,a = p[0]
		if type(x) == np.ndarray:
			f = []
			for el in x:
				if y_s > el*a:
					f+=[np.float(el*a)]
				else:
					f+=[np.float(y_s)]
		else:
			if y_s > x*a:
				f = [np.float(x*a)]
			else:
				f = [np.float(y_s)]
		return np.array(f)

	def fit_knee(x,x_k):
		f = []
		if type(x) == np.ndarray:
			for el in x:
				if el<x_k:
					f+=[np.float(el)]
				else:
					f+=[np.float(x_k)]
		else:
			if x<x_k:
				f+=[np.float(x)]
			else:
				f+=[np.float(x_k)]
		return np.array(f)

	##Dictionary containing the fitting functions, and the number of parameters required for each
	func_list = dict([('fit_line',[fit_line,1]),('fit_line_offset',[fit_line_offset,2]),('fit_pl2',[fit_pl2,2]),('fit_pl3',[fit_pl3,3]),('fit_pl4',[fit_pl4,4]),('fit_lin2',[fit_lin2,2]),('fit_lin3',[fit_lin3,3]),('fit_pl_slopes',[fit_pl_slopes,4]),('fit_flat',[fit_flat,2]),('fit_flat_inv',[fit_flat_inv,2]),('fit_knee',[fit_knee,1])])
	###################################################################################

	def load_data(self,path_to_data,names=None):
		"""Load data to be fitted. Argument of the function should be the full path to the data. If a list on names is included, it must have the same ordering as the columns of the data file"""
		self.data = np.loadtxt(path_to_data)
		if names: 
			self.namelist = np.loadtxt(names,usecols=0,dtype='str')
			for ii in range(len(self.namelist)):
				if self.namelist[ii][0]=='/': self.namelist[ii]=self.namelist[ii][1:]
		for ii in reversed(range(len(self.data))):
			if np.isnan(self.data[ii][2]) or np.isinf(self.data[ii][2]) or np.isnan(self.data[ii][3]) or np.isinf(self.data[ii][3]):
				self.data = np.delete(self.data,ii,0)
				if names: self.namelist=np.delete(self.namelist,ii,0)
		self.Nspec = len(self.data)
		self.data = self.data.T

	def make_catalogue_file(self,catalogue,outfile='sample_catalogue_data',use_sdss_name=False):
		'''Load the data from a quasar catalog file (format Shen et al. DR7), and store in a file. Note that all objects come in pairs, so one set of characteristics is required per pair of spectra.'''
		try: hdu = fits.open(catalogue); T=hdu[1].data; 
		except: print 'Please specify the catalogue file (Shen et al DR7)'; return
		spec_list = []
		for ii in range(len(self.namelist)):
			sp = self.namelist[ii].split('-')
			spec_list += [(self.namelist[ii],sp[1],sp[2],sp[3][:-5],self.data[4][ii])]
		spec_list = np.array(spec_list,dtype=[('name','S25'),('plate','int'),('mjd','int'),('fiber','int'),('z',float)])
		D = []
		m = 0
		if use_sdss_name==False:
			spec_list = np.sort(spec_list,order=['plate','mjd','fiber'])
			print 'starting with copy'
			for ii in range(len(T['plate'])):
				D += [(T['plate'][ii],T['mjd'][ii],T['fiber'][ii],T['sdss_name'][ii],T['logl3000'][ii],T['loglbol'][ii],T['logbh'][ii],T['logedd_ratio'][ii],T['r_6cm_2500a'][ii],T['ew_mgii'][ii])]
			print 'copied relevant columns'
			T = np.array(D,dtype=[('plate','int'),('mjd','int'),('fiber','int'),('sdss_name','S25'),('logl3000',float),('loglbol',float),('logbh',float),('logedd_ratio',float),('r_6cm_2500a',float),('ew_mgii',float)])
			del D; hdu.close()
			print 'sorting rows'
			T = np.sort(T,order=['plate','mjd','fiber'])
			print 'sorted'
		lc = []
		start,jj=0,0
		for obj in spec_list:
			if jj%100==0:
				print jj, obj['name']
			jj+=1
			if obj['mjd']>55000: continue
			for ii in range(start,len(T['plate'])):
				if T['plate'][ii]==obj['plate'] and T['fiber'][ii]==obj['fiber'] and T['mjd'][ii]==obj['mjd']:
					start = ii
					lc +=[[obj['name'],T['sdss_name'][ii],obj['z'],T['logl3000'][ii],T['loglbol'][ii],T['logbh'][ii],T['logedd_ratio'][ii],T['r_6cm_2500a'][ii],T['ew_mgii'][ii]]]
					break
		with open(outfile,'w') as of:
			of.write('#{:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}\n'.format('spec_name','sdss_name','z','logl3000','loglbol','logbh','logEdd','r_loud','ew_mgii'))
			for r in lc:
				of.write(' {:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25} {:>25}\n'.format(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8]))

	def filter_data(self,delta=False,name_pair=False,norm_clean_max=False,norm_clean_min=False,rm_delsig=False):
		"""Remove spectra with NaN values in their normalised fit results, as well as the spectra used for normalisation. Different keywords are used for different normalisations."""
		self.data = self.data.T
		N_spec = len(self.data)
		nan_count = 0; del_lst =[]
		if name_pair==True:
			for ii in range(0,len(self.namelist)-1,2):
				if int(self.namelist[ii].split('-')[2])<int(self.namelist[ii+1].split('-')[2]):
					self.namelist[ii+1]=self.namelist[ii]
				else:
					self.namelist[ii]=self.namelist[ii+1]
		#remove Nan
		for ii in reversed(range(N_spec)):
			if np.count_nonzero(np.isnan(self.data[ii]))!=0:
				nan_count += 1
				del_lst += [ii]
		#Clean normalised data
		if norm_clean_max==True:
			for ii in range(len(self.data)):
				if np.count_nonzero(np.isinf(self.data[ii]))!=0 or self.data[ii][0]<0.01 or self.data[ii][0]>1 or self.data[ii][2]<0 or self.data[ii][2]>2:
					del_lst+=[ii]
		elif norm_clean_min==True:
			for ii in range(len(self.data)):
				if np.count_nonzero(np.isinf(self.data[ii]))!=0 or self.data[ii][0]<1 or self.data[ii][0]>20 or self.data[ii][2]<0.0 or self.data[ii][2]>15:
					del_lst+=[ii]
		print 'Number of quality-removed spectra = {} out of {}'.format(nan_count,N_spec)
		if rm_delsig==True:
			for ii in range(len(self.data)):
				if self.data[ii][8]==0: del_lst+=[ii]
		#Remove objects with the same value of delta_t, and all entries normalised to one
		if delta==True:
			delta = self.data[0][6]
			for ii in reversed(range(N_spec)):
				if self.data[ii][6] == delta or self.data[ii][0]==1: del_lst+=[ii]
				else: delta = self.data[ii][6]
		else:
			for ii in reversed(range(N_spec)):
				if self.data[ii][0]==1.: del_lst+=[ii]
		del_lst=np.unique(del_lst)
		self.data = np.delete(self.data,del_lst,axis=0)
		if len(self.namelist)!=0: self.namelist=np.delete(self.namelist,del_lst,axis=0)
		self.Nspec = len(self.data)
		print 'Number of spectra in final data-set = {}'.format(self.Nspec)
		self.data = self.data.T

	def find_spearman(self,prnt=False):
		'''Calculate the p-value for t-statistic derived from the Spearman correlation coefficient'''
		r1 = stat.rankdata(self.data[0])
		r2 = stat.rankdata(self.data[2])
		bn = len(self.data[0])
		self.scc = 1 - 6./(bn*(bn*bn-1))*sum(np.power(r2-r1,2))
		if prnt:
			print 'Spearman correlation coefficient = {}'.format(self.scc)

	def find_pearson_pval(self,prnt=False,in1=[],in2=[]):
		'''Calculate the p-value for t-statistic derived from the Spearman correlation coefficient'''
		if len(in1)==0: in1=self.data[0]
		if len(in2)!=len(in1): in2=self.data[2]
		a1 = np.average(in1); s1 = np.std(in1)
		a2 = np.average(in2); s2 = np.std(in2)
		cov = np.sum((in1-a1)*(in2-a2))/len(in1)
		self.pcc = cov/(s1*s2)
		TPcc = self.pcc*np.sqrt((len(in1)-2)/(1-self.pcc*self.pcc))
		self.pcc_p = 2*stat.t.sf(TPcc, len(in1)-2)
		if prnt:
			print 'Pearson correlation coefficient = {}, with a p-value for the associated t-statistic of {}'.format(self.pcc,self.pcc_p)

	def do_fit(self,**kwargs):
		"""Call the scipy.optimize.curve_fit function, in combination with the preset fitting function and the loaded data to perform a fit on the data"""
		self.fit_func, self.npars = self.func_list[self.fname]
		self.par, self.pcov = [],[]
		if 'fit_data' in kwargs:
			fit_data = kwargs['fit_data']
		else:
			fit_data = self.data
		if 'sig' in kwargs: sig=kwargs['sig']
		else: sig = np.full(len(fit_data[0]),1)
		if 'bnds' in kwargs:
			self.par, self.pcov = curve_fit(self.fit_func,fit_data[0],fit_data[2],p0=np.full(self.npars,1),sigma=sig,bounds=kwargs['bnds'])
		else: self.par, self.pcov = curve_fit(self.fit_func,fit_data[0],fit_data[2],p0=np.full(self.npars,1),sigma=sig)

	def find_residuals(self,fit_data=[]):
		'''Find the residuals with respect to the defined fitting function'''
		if len(fit_data)==0: fit_data = self.data
		self.residuals = fit_data[2]-self.fit_func(fit_data[0],*self.par)

	def find_chi2(self,sig):
		'''return chi^2 using the residuals of the fit. Requires the array of errors as input'''
		try: len(self.residuals)
		except:	self.find_residuals()
		self.chi2 = np.sum(np.power(self.residuals/sig,2))
		self.red_chi2 = np.sum(np.power(self.residuals/sig,2))/(len(self.data[0])-len(self.par))

	def sequence_test(self,inp_array=[],prnt=True):
		if len(inp_array)==0: inp_array=self.residuals
		S,D = 0,0
		for ii in range(1,len(inp_array)):
			if inp_array[ii-1]*inp_array[ii]>0:
				S+=1
			else: D+=1
		pval=1-stat.binom_test([S,D],p=0.5)
		if prnt==True: print 'Number of sequential data points = {}, number of different points = {}, with a p-value of {}'.format(S,D,pval)
		return float(S),pval

	def compare_models(self,fit_data=[],fname=['fit_line','fit_flat_inv'],prnt=True):
		'''Fit two models to the data (iteratively) and calculate comparative goodness of fit statistics (delta chi2 and sequence test)\n
		Note that for the initial fit, using the lmfit errors, all zeros are filtered out and replaced with an average error
		NOTE: different use of 'sig' in fitting algorithm, to ensure the fits to be compared use the same errors (CHECK THIS)\n
		fname needs to be an iterable containing the name(s) of the function(s) to be fitted'''
		if len(fit_data)==0: fit_data=self.data
		a = np.average(fit_data[3][np.where(fit_data[3]!=0)])
		for jj in range(len(fit_data[3])):
			if fit_data[3][jj]==0: fit_data[3][jj]=a
		chi2,red_chi2,seq,residuals,pars,errs = [],[],[],[],[],[]
		for ii in range(len(fname)):
			self.fname = fname[ii]
			self.do_fit(fit_data=fit_data,sig=fit_data[3])
			self.find_residuals(fit_data=fit_data)
			if 'sig' not in locals():
				sig = np.full(len(self.residuals),np.std(self.residuals))
			self.do_fit(fit_data=fit_data,sig=sig)
			self.find_residuals(fit_data=fit_data)
			self.find_chi2(sig=sig)
			chi2 += [self.chi2]; red_chi2 += [self.red_chi2]
			seq += [self.sequence_test(prnt=False)]
			residuals += [self.residuals]; pars += [self.par]; errs += [np.sqrt(np.diag(self.pcov))]
		if len(fname)!=1: d_chi2 = abs(chi2[0]-chi2[1])
		else: d_chi2=0
		if prnt==True:
			print 'For the models {}  and {}, the following test statistics apply:'.format(fname[0],fname[1])
			print 'For {}, chi2 = {}; for {}, chi2 = {}'.format(fname[0],round(chi2[0],5),fname[1],round(chi2[1],5))
			print 'Delta_chi2 = {}, with an associated p-value of {}'.format(round(d_chi2,5),round(stat.chi2.sf(d_chi2,1),3))
			print 'For {}, p_s = {}; for {}, p_s = {}'.format(fname[0],round(seq[0][1],5),fname[1],round(seq[1][1],5))
		for ii in range(len(seq)): seq[ii]=seq[ii][0]
		return dict([('fname',fname),('d_chi2',d_chi2),('red_chi2',tuple(red_chi2)),('seq',tuple(seq)),('residuals',residuals),('parameters',pars),('fit_err',errs)])

	def bootstrap_mc_cmp(self,fname=['fit_line','fit_flat_inv'],Nmc=10):
		'''Repeat the model comparison on a bootstrapped sample of the data, repeating this Nmc times in a Monte Carlo experiment'''
		indx = np.random.randint(0,len(self.data[0]),size=Nmc)
		outp = []
		for ii in range(Nmc):
			masked_data = np.delete(self.data.T,indx[ii],axis=0).T
			cm = self.compare_models(fit_data=masked_data,fname=fname,prnt=False)
			outp_row = [cm['d_chi2']]
			for jj in range(len(fname)): outp_row+=[cm['seq'][jj]]
			outp += [outp_row]
		outp = np.array(outp).T
		a_chi2 = np.average(outp[0])
		print 'Average Delta chi2 = {} (+-{}), giving a p-value of {}'.format(round(a_chi2,3),round(np.std(outp[1]),3),round(stat.chi2.sf(a_chi2,1),3))
		for ii in range(len(fname)):
			print 'For {}, average S={}+/-{}'.format(fname[ii],round(np.average(outp[1+ii]),2),round(np.std(outp[1+ii])))

	def linregress_lvsc(self,val='con_line',sample='fps',lim=5):
		'''Linear regression of either the log10(line) on the log10(continuum) fluxes, or of the log10(sigma) on the log10(continuum).'''
		d_corr = 4*np.pi*np.power(cosmo.comoving_distance(self.data[4]).cgs.value*(1.+self.data[4]),2)
		if val=='con_line':
			X=np.log10(self.data[0]*d_corr*2798)#; Xerr=self.data[13]*d_corr*2798*np.log10(np.e)/self.data[0]
			Y=np.log10(self.data[2]*d_corr); Yerr=self.data[15]*d_corr*np.log10(np.e)/self.data[2]
		elif val=='con_sig':
			X=np.log10(self.data[0]*d_corr*2798)#; Xerr=self.data[13]*d_corr*2798*np.log10(np.e)/self.data[0]
			Y=np.log10(self.data[8]/(1+self.data[4])); Yerr=self.data[15]/(1+self.data[4])*np.log10(np.e)/self.data[8]
		dX,dY,dYerr=[],[],[]
		if sample=='fps':#Full Population sample data format
			print 'sample=fps: assuming all objects come in pairs'
			for ii in range(0,len(self.data[0]),2):
				dx=X[ii+1]-X[ii]
				if abs(dx)>lim: continue
				dy=Y[ii+1]-Y[ii]
				if abs(dy)>lim: continue
				dX+=[dx]; dY+=[dy]; dYerr+=[np.sqrt(Yerr[ii]*Yerr[ii]+Yerr[ii+1]*Yerr[ii+1])]
		elif sample=='svs':#Supervariable sample data format
			for ii in reversed(range(1,len(self.namelist))):
				if self.namelist[ii]==self.namelist[ii-1]:
					dx=X[ii+1]-X[ii]
					if abs(dx)>lim: continue
					dy=Y[ii+1]-Y[ii]
					if abs(dy)>lim: continue
					dX+=[dx]; dY+=[dy]; dYerr+=[np.sqrt(Yerr[ii]*Yerr[ii]+Yerr[ii+1]*Yerr[ii+1])]
		else: print 'please select a correct sample name'; return
		dx=X[1]-X[0]; dy=Y[1]-Y[0]
		if abs(dx)>lim and abs(dy)>lim: dX+=[dx]; dY+=[dy]; dYerr+=[np.sqrt(Yerr[0]*Yerr[0]+Yerr[1]*Yerr[1])]
		dX=np.array(dX); dY=np.array(dY); dYerr=np.array(dYerr)
		dY=dY[np.isnan(dX)==0]
		dYerr=dYerr[np.isnan(dX)==0]
		dX=dX[np.isnan(dX)==0]
		dX=dX[np.isnan(dY)==0]
		dYerr=dYerr[np.isnan(dY)==0]
		dY=dY[np.isnan(dY)==0]
		dY=dY[np.isinf(dX)==0]
		dYerr=dYerr[np.isinf(dX)==0]
		dX=dX[np.isinf(dX)==0]
		dX=dX[np.isinf(dY)==0]
		dYerr=dYerr[np.isinf(dY)==0]
		dY=dY[np.isinf(dY)==0]
		print len(X),len(Y),len(Yerr)
		print len(dX),len(dY),len(dYerr)
		print np.amin(dX),np.amax(dX)
		print np.amin(dY),np.amax(dY)
		a = stat.linregress(dX,dY)
		self.fname='fit_line'
		self.fit_func, self.npars = self.func_list[self.fname]
		self.par, self.pcov = curve_fit(self.fit_func,dX,dY,p0=np.full(self.npars,1))
		print a
		print self.par,self.pcov

	def create_figure(self,fig_type='residuals',saveloc='Figures/',name='MgII_figure.pdf',s7_data='False',catalogue_file=None,cat_cmp='',fit_data=[],mod_cmp=[],lims=[0,1.05,0,1.75],nbins=30,nbins_sub=10,zbins=[.5,1,1.5,5],dtbins=[3,5,7,9],levels_pct=[.25,.5,.75,.9,.95],dt_rng=None,boxs=None,additional_data=[],contours=False,xlog=False,ylog=False,log_counts=False,rm_lims=((-100,100),),lum=None,cmap='RdBu',absval=False,rm_cat=False, clq_namelist=None ):
		'''Create the figures; the figure type is selected with the keyword 'fig_type'. The following is list of the available types.\n
		-1- residuals: linear fits to the data (of the Supervariable sample) and the residuals (F5&6)\n
		-2- lum_overview: change in line luminosity plotted against change in continuum luminosity. Required file format: C Ce L Le z z_sdss dt S/N dC dCe dL dL, where the Continuum (C) is measured at 2798 AA (F4)\n
		-3- dt_histogram: histogram of the Delta t values in the sample(s) (F1)\n
		-3- colourmap: 2D histogram of normalised line and continuum fluxes (F5&6)\n
		-4- dtbin_colourmap: four panels containing the same histograms as 'colourmap', split by Delta t (F12)\n
		-5- subsample_catalogue: sample distribution of Lbol,Mbh, and Eddington ratio, based on subsections in normalised flux space (F20)\n
		-6- response_catalogue: sample distribution of Lbol,Mbh, and Eddington ratio, based on the values of alpha_rm (F21)\n
		-7- flux_vs_dt: create subsamples based on Delta t of the relative change in flux (F13)\n
		-8- continuum_vs_dt: as flux_vs_dt for continuum flux levels\n
		-9- response_metric_hist: histogram of alpha_rm (F7)\n
		-10- cmp_clq: compare CLQ and non-CLQ identifications in the Supervariable sample (F8)\n
		-11- del_sigma_hist: histogram of Delta sigma (F14)\n
		-12- sig_vs_con: plot sigma against the continuum flux in a 2D histogram\n
		-13- delsig_vs_delcon: plot Delta sigma against Delta f_c in a 2D histogram\n
		-14- sig_vs_line: plot sigma against the line flux (F16)\n
		-15- delsig_vs_delline: plot Delta sigma against Delta f_MgII (F15)\n
		-16- cvsf_individual: plot the normalised line flux against the normalised continuum for J022556 (F10)\n
		-17- ew_histogram: histogram of Equivalent Widths (F11)\n
		-18- lzedd: Lbol plotted against redshift, colour-coded by Eddinton ratio (F19)\n
		'''
		if fig_type=='residuals':
			if len(fit_data)==0: fit_data=self.data
			if len(mod_cmp)==0: mod_cmp = dict([('fname',self.fname),('d_chi2',None),('red_chi2',0),('seq',None),('residuals',self.residuals),('parameters',self.par)])
			fig=plt.figure(figsize=(6,5))
			h = 4+2*len(mod_cmp['fname'])
			gs=gridspec.GridSpec(h,1)
			ax0 = fig.add_subplot(gs[0:4,0])
			axs = []
			for ii in range(len(mod_cmp['fname'])):
				hs,he = 4+2*ii,4+2*(ii+1)
				axs += [fig.add_subplot(gs[hs:he,0])]
			plt_rng = np.arange(lims[0],lims[1],0.01)
			fclr = ['r','darkblue']; eclr=['r','darkblue']; ls = ['-','-.']
			ax0.axis(lims)
			ax0.set_yticks(np.arange(1,6.1,step=1))
			ax0.tick_params(axis='x',direction='in',which='both')
			ax0.xaxis.set_ticklabels([])
			ax0.scatter(fit_data[0],fit_data[2],marker='o',c='g',edgecolor='k')
			for ii in range(len(fit_data[0])):
				if fit_data[0][ii]>10: ax0.scatter(fit_data[0][ii],fit_data[2][ii],marker='s',c='darkorchid',edgecolor='k')
				if fit_data[2][ii]>5.5: ax0.scatter(fit_data[0][ii],fit_data[2][ii],marker='s',c='cyan',edgecolor='k')
			for ii in range(len(mod_cmp['fname'])):
				ax0.plot(plt_rng,self.func_list[mod_cmp['fname'][ii]][0](plt_rng,*mod_cmp['parameters'][ii]),c=fclr[ii],ls=ls[ii])
				axs[ii].scatter(fit_data[0],mod_cmp['residuals'][ii],marker='o',facecolor=fclr[ii],edgecolor=eclr[ii],zorder=100)
				axs[ii].axis([lims[0],lims[1],-.5,.7])
				axs[ii].set_yticks([-4,-2,0,2,4])
				axs[ii].axhline(y=0,xmin=lims[0],xmax=lims[1],ls='--',c='grey')
				if mod_cmp['red_chi2']==0:
					if self.red_chi2:
						ax0.text(0.75,0.1,r'$\chi_{red}^2$ = '+str(round(self.red_chi2,2)),transform=ax0.transAxes)
				else:
						axs[ii].text(0.05,0.1,r'$\chi_{red}^2$ = '+str(round(mod_cmp['red_chi2'][ii],3)),transform=axs[ii].transAxes)
				axs[ii].minorticks_on()
				axs[ii].tick_params(labelbottom=0)
				axs[0].text(0.9,0.8,'1C',transform=axs[0].transAxes,fontweight='bold',fontsize=13)
				axs[1].text(0.9,0.8,'2C',transform=axs[1].transAxes,fontweight='bold',fontsize=13)
			ax0.minorticks_on()
			plt.xlabel(r'2798$\rm\AA$ Continuum Flux',fontsize=14,fontweight='bold')
			ax0.set_ylabel('MgII Line Flux',fontsize=14,fontweight='bold')
			plt.text(0.029,0.38,'Residuals',fontsize=14,fontweight='bold',rotation=90,transform=fig.transFigure)
			axs[-1].tick_params(labelbottom=1)
			fig.subplots_adjust(hspace=0)
		elif fig_type=='lum_overview':
			fig,ax=plt.subplots(1,figsize=(7,7))
			d_corr = 4*np.pi*np.power(cosmo.comoving_distance(self.data[4]).cgs.value*(1.+self.data[4]),2)
			self.data[12]=np.absolute(self.data[12])*d_corr*2798; self.data[13]=self.data[13]*d_corr*2798
			self.data[14]=np.absolute(self.data[14])*d_corr; self.data[15]=self.data[15]*d_corr
			err_est_dc = np.average(self.data[13][self.data[13]!=0])
			err_est_dl = np.average(self.data[15][self.data[15]!=0])
			for ii in range(len(self.data[0])):
				if self.data[13][ii]==0:
					self.data[13][ii]=err_est_dc
				if self.data[15][ii]==0:
					self.data[15][ii]=err_est_dl
			fd = np.log10(self.data[[12,13,14,15]])
			fd[1]=np.log10(np.e)*self.data[13]/self.data[12]; fd[3]=np.log10(np.e)*self.data[15]/self.data[14]
			ax.errorbar(fd[0],fd[2],xerr=fd[1],yerr=fd[3],fmt='o',c='navy',markersize=8,ecolor='grey',capsize=2)
			lims = [40,45.5,40,43.5]
			a = stat.linregress(fd[0],fd[2]); print a; ax.plot(np.linspace(lims[0],lims[1],10),a[0]*np.linspace(lims[0],lims[1],10)+a[1],c='r',linestyle='--',zorder=5)
			ax.axis(lims)
			ax.set_xlabel(r'$\mathrm{log}_{10}|\Delta\lambda\mathrm{L}_{2798}|$ (erg s$^{-1}$)',fontsize=18,fontweight='bold')
			ax.set_ylabel(r'$\mathrm{log}_{10}|\Delta\mathrm{L}_{\mathrm{MgII}}|$ (erg s$^{-1}$)',fontsize=18,fontweight='bold')
			ax.tick_params(axis='both',labelsize=16)
			plt.tight_layout()
		elif fig_type=='dt_histogram':
			fig,ax = plt.subplots(1,figsize=(6,4))
			plt.grid(True,ls='dotted',zorder=1)
			counts,bins,image = plt.hist(self.data[6]/(1.+self.data[4]),nbins,facecolor='r',alpha=0.85,edgecolor='r',zorder=9,label='Full Population')
			if len(additional_data)!=0:
				ad = additional_data
				counts,bins,image = plt.hist(ad[6]/(1.+ad[4]),nbins,facecolor='b',alpha=0.6,zorder=9,label='Supervariable',edgecolor='k')
			ax.set_xlabel(r'$\Delta$t (days in QSO restframe)',fontweight='bold',fontsize=16)
			ax.set_ylabel(r'N$_{\mathrm{QSO}}$',fontweight='bold',fontsize=16)
			if ylog==True:
				ax.set_yscale('log')
			ax.set_xlim(0,np.amax(bins))
			ax.legend(loc='upper right',edgecolor='k')
			plt.tight_layout()
		elif fig_type=='colourmap':
			fig,ax = plt.subplots(1,figsize=(7,6))
			xstretch = (lims[3]-lims[2])/float(lims[1]-lims[0])
			counts,xb,yb,image = plt.hist2d(xstretch*self.data[0],self.data[2],bins=nbins,range=[[xstretch*lims[0],xstretch*lims[1]],lims[2:]])
			m = np.amax(counts)
			if log_counts==True:
				norm = cm.colors.LogNorm()
			else: norm = cm.colors.Normalize(0,m)
			levels_pct = np.array(levels_pct)*m
			plt.cla()
			extent = [xb[0],xb[-1],yb[0],yb[-1]]
			im = ax.imshow(counts.T,extent=extent,cmap=cmap,origin='lower',norm=norm)
			ax.contour(counts.T,extent=extent,levels=levels_pct)
			cax = fig.add_axes([0.88, 0.13, 0.02, 0.65])
			fig.colorbar(im,cax,orientation='vertical')
			lbl = np.round(np.arange(lims[0],lims[1]+.01,round((lims[1]-lims[0])/4.,2)),2)
			if xlog==True: ax.set_xscale('log')
			if ylog==True: ax.set_yscale('log')
			ax.set_xticks(np.linspace(xb[0],xb[-1],len(lbl)))
			ax.set_xticklabels(lbl.astype('str'))
			ax.set_xlabel(r'2798$\rm\AA$ Continuum Flux',fontsize=17,fontweight='bold')
			ax.set_ylabel('MgII Line Flux',fontsize=17,fontweight='bold')
			cax.text(-1.8,1.02,'Counts',transform=cax.transAxes,fontweight='bold',fontsize=15)
		elif fig_type=='dtbin_colourmap':
			fig = plt.figure(figsize=(6,6))
			gs = gridspec.GridSpec(4,4)
			gs.update(wspace=0., hspace=-0.)
			axs = [fig.add_subplot(gs[0:2,0:2]),fig.add_subplot(gs[0:2,2:]),fig.add_subplot(gs[2:,0:2]),fig.add_subplot(gs[2:,2:])]
			for ii in range(4):
				dtbins[ii] = 365.25*dtbins[ii]
			dtbins = [0]+dtbins
			box = dict(boxstyle='square',facecolor='w')
			dtz_arr = self.data[6]/(self.data[5]+1)
			for ii in range(4):
				axs[ii].axis(lims)
				args = np.argwhere( (dtz_arr > dtbins[ii]) & (dtz_arr < dtbins[ii+1]) )
				counts,xb,yb,image = plt.hist2d(self.data[0][args].flatten(),self.data[2][args].flatten(),bins=nbins,range=[lims[:2],lims[2:]])
				m = np.amax(counts); counts=counts/float(m)
				extent = [xb[0],xb[-1],yb[0],yb[-1]]
				im = axs[ii].imshow(counts.T,aspect='auto',extent=extent,cmap='YlOrRd',origin='lower')
				if ii==0: save_counts=counts; save_ext=extent
				if ii==3: axs[ii].contour(save_counts.T,extent=save_ext,levels=levels_pct,linestyles=':')
				axs[ii].contour(counts.T,extent=extent,levels=levels_pct)
				axs[ii].text(.1,.05,r'{}<$\Delta$t<{} N={}'.format(int(dtbins[ii]),int(dtbins[ii+1]),len(self.data[0][args])),transform=axs[ii].transAxes,bbox=box)
				mod_cmp=self.compare_models(fname=['fit_line'],prnt=False,fit_data=[self.data[0][args].flatten(),self.data[1][args].flatten(),self.data[2][args].flatten(),self.data[3][args].flatten()])
				print 'dt_bin =',ii,'slope =',mod_cmp['parameters'],'fit_err =',mod_cmp['fit_err']
			cax = fig.add_axes([0.92, 0.15, 0.02, 0.67])
			fig.colorbar(im,cax,orientation='vertical')
			fig.text(.52,.04,r'2798$\rm\AA$ Continuum Flux',fontsize=15,fontweight='bold',ha='center')
			fig.text(.02,.5,'MgII Line Flux',fontsize=15,fontweight='bold',va='center',rotation='vertical')
			axs[0].tick_params(direction='out',top=1,labeltop=0,labelbottom=0,labelleft=0)
			axs[1].tick_params(direction='out',top=1,labeltop=0,right=1,labelright=0,labelleft=0,labelbottom=0)
			axs[3].tick_params(direction='out',right=1,labelright=0,labelleft=0,labelbottom=0)
		elif fig_type=='subsample_catalogue':
			try:
				col_names = ('spec_name','sdss_name','z','logl3000','logbol','logbh','logEdd','rloud','ewmgii')
				formats = ('S25','S30')+7*('float',)
				self.catalogue = np.loadtxt(catalogue_file,dtype={'names':col_names,'formats':formats})
			except: print 'Please create catalogue_file using make_catalogue_file'; return
			fig = plt.figure(figsize=(12,3))
			gs = gridspec.GridSpec(1,4)
			gs.update(wspace=0.,hspace=0.)
			axs = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[0,3])]
			fclr = ['g','b','None']; eclr = ['g','b','k']; al=[.5,.5,1];lbl=['a','b','c','d']
			xstretch = (lims[3]-lims[2])/float(lims[1]-lims[0])
			counts,xb,yb,image = plt.hist2d(xstretch*self.data[0],self.data[2],bins=nbins,range=[[xstretch*lims[0],xstretch*lims[1]],lims[2:]])
			if log_counts==True:
				norm = cm.colors.LogNorm()
			else: norm = cm.colors.Normalize(0,np.amax(counts))
			extent = [xb[0],xb[-1],yb[0],yb[-1]]
			plt.cla()
			im = axs[0].imshow(counts.T,extent=extent,cmap=cmap,origin='lower',norm=norm)
			axs[0].plot(xstretch*np.linspace(lims[0],lims[1],10),np.linspace(lims[0],lims[1],10),c='k',ls='--')
			xlim,ylim = [],[]; mm=0
			for box in boxs:
				for ii in range(len(xb)):
					if xb[ii]>xstretch*box[0][0]: break
				for jj in range(len(yb)):
					if yb[jj]>box[0][1]: break
				for kk in reversed(range(len(xb))):
					if xb[kk]<xstretch*box[1][0]: break
				for ll in reversed(range(len(yb))):
					if yb[ll]<box[1][1]: break
				xlim+=[(ii-1,kk+1)]
				ylim+=[(jj-1,ll+1)]
				SQ = mpt.Rectangle((xb[ii-1],yb[jj-1]),(xb[kk+1]-xb[ii-1]),yb[ll+1]-yb[jj-1],linewidth=2,edgecolor=eclr[mm],fill=False)
				axs[0].add_patch(SQ)
				bwx = (lims[1]-lims[0])/float(nbins); bwy = (lims[-1]-lims[-2])/float(nbins)
				axs[0].text(xb[ii]+.2*bwx,yb[ll]-1*bwy,lbl[mm],color=eclr[mm],fontsize=14)
				mm+=1
			dist_ex = [[100,0,0],[100,0,0],[100,0,0]]
			for mm in range(len(boxs)):
				xl,xu = xlim[mm][0],xlim[mm][1]; yl,yu = ylim[mm][0],ylim[mm][1]
				dist=[[],[],[]]
				total, missing = 0,0
				for pp in range(xl,xu):
					for qq in range(yl,yu):
						names=[]
						for ii in range(len(self.data[0])):
							if (xstretch*self.data[0][ii])>xb[pp] and (xstretch*self.data[0][ii])<xb[pp+1]:
								if self.data[2][ii]>yb[qq] and self.data[2][ii]<yb[qq+1]:
									names+=[self.namelist[ii]]
						for obj in names:
							total+=1
							ind = np.where(self.catalogue['spec_name']==obj)
							if len(ind[0])!=0:
								dist[2] += [ self.catalogue['logEdd'][ind][0] ]
								dist[1] += [ self.catalogue['logbh'][ind][0] ]
								dist[0] += [ self.catalogue['logbol'][ind][0] ]
							else: missing+=1
				print 'total =', total,'; missing =', missing
				for ii in reversed(range(len(dist[0]))):
					if dist[0][ii]<40 or dist[0][ii]>50:
						del dist[0][ii]
					if dist[1][ii]<5 or dist[1][ii]>15:
						del dist[1][ii]
					if dist[2][ii]<-3 or dist[2][ii]>2:
						del dist[2][ii]
				for ii in range(len(dist)):
					if ii==0: lbl_ = lbl[mm]+': N={}'.format(len(dist[ii]))
					else: lbl_=lbl[mm]
					c,b,i = axs[ii+1].hist(dist[ii],bins=nbins_sub,facecolor=fclr[mm],edgecolor=eclr[mm],alpha=al[mm],density=True,label=lbl_)
					if np.amin(dist[ii])<dist_ex[ii][0]: dist_ex[ii][0]=np.amin(dist[ii])
					if np.amax(dist[ii])>dist_ex[ii][1]: dist_ex[ii][1]=np.amax(dist[ii])
					if np.amax(c)>dist_ex[ii][2]: dist_ex[ii][2]=np.amax(c)
					print '###########################\nbox {}; dist {}'.format(lbl[mm],ii)
					print 'N={} ; Median={}; Avg={}; Std={}'.format( len(dist[ii]),np.median(dist[ii]),np.average(dist[ii]),np.std(dist[ii]) )
					axs[ii+1].axis([dist_ex[ii][0],dist_ex[ii][1],0,1.2*dist_ex[ii][2]])
					axs[ii+1].legend(loc='upper left',numpoints=1, edgecolor='k')
			for ii in range(len(dist)):
				axs[ii+1].legend(loc='upper left',numpoints=1, edgecolor='k')
				axs[ii+1].tick_params(labelleft=0,right=1,direction='in')
			lbl = np.round(np.arange(lims[0],lims[1]+.01,round((lims[1]-lims[0])/4.,2)),2)
			axs[0].set_xticks(np.linspace(xb[0],xb[-1],len(lbl)))
			axs[0].set_xticklabels(lbl.astype('str'))
			axs[0].set_xlabel('Continuum Flux',fontsize=14,fontweight='bold')
			axs[0].set_ylabel('MgII Line Flux',fontsize=14,fontweight='bold')
			axs[1].set_xlabel(r'log$_{10}$ L$_{bol}$',fontsize=14,fontweight='bold')
			axs[2].set_xlabel(r'log$_{10}$ M$_{BH}$',fontsize=14,fontweight='bold')
			axs[3].set_xlabel(r'log$_{10}$ L$_{bol}$/L$_{Edd}$',fontsize=14,fontweight='bold')
			axs[3].set_ylabel(r'Normalised N$_{spec}$',fontsize=14,fontweight='bold')
			axs[3].yaxis.set_label_position("right")
			plt.tight_layout()
		elif fig_type=='response_catalogue':
			try:
				col_names = ('spec_name','sdss_name','z','logl3000','logbol','logbh','logEdd','rloud','ewmgii')
				formats = ('S30','S30')+7*('float',)
				self.catalogue = np.loadtxt(catalogue_file,dtype={'names':col_names,'formats':formats})
				if cat_cmp:
					cat_cmp = np.loadtxt(cat_cmp,dtype={'names':col_names,'formats':formats}).T
			except: print 'Please create catalogue_file using make_catalogue_file'; return
			if rm_cat==1:
				names = []
				for lim in rm_lims:
					names += [[]]
				for ii in range(len(self.data[0])):
					rm = self.data[2][ii]-self.data[0][ii]
					if abs(rm)<10:
						for jj in range(len(rm_lims)):
							if rm>rm_lims[jj][0] and rm<rm_lims[jj][1]: names[jj]+=[self.namelist[ii]]
				lbl=[r'$\alpha <${}'.format(rm_lims[0][1]),r'$\alpha >$ {}'.format(rm_lims[1][0])]
			else:
				names=[self.namelist]
				lbl=['Full Population','Supervariable']
			fig,ax = plt.subplots(2,2,figsize=(8,8))
			axs=[ax[0,0],ax[0,1],ax[1,0],ax[1,1]]
			fig.subplots_adjust(wspace=0)
			clr = ['r','b']
			total, missing = 0,0
			dist_ex=[[100,0,0],[100,0,0],[100,0,0],[100,0,0]] #Used to control the axis in the plot
			for jj in range(len(names)):
				dist=[[],[],[],[]]
				#Create the distributions
				for obj in names[jj]:
					total+=1
					ind = np.where(self.catalogue['spec_name']==obj)
					if len(ind[0])!=0:
						dist[3] += [ self.catalogue['z'][ind][0] ]
						dist[2] += [ self.catalogue['logEdd'][ind][0] ]
						dist[1] += [ self.catalogue['logbh'][ind][0] ]
						dist[0] += [ self.catalogue['logbol'][ind][0] ]	
					else: missing+=1
				#Filter out erroneous values
				for ii in reversed(range(len(dist[0]))):
					if dist[0][ii]<40 or dist[0][ii]>50:
						del dist[0][ii]
					if dist[1][ii]<5 or dist[1][ii]>15:
						del dist[1][ii]
					if dist[2][ii]<-3 or dist[2][ii]>2:
						del dist[2][ii]
					if dist[3][ii]<0 or dist[3][ii]>4:
						del dist[3][ii]
				#Create the plots
				for ii in range(len(dist_ex)):
					if rm_cat==1 and ii==3: lbl_=lbl[jj]+': N={}'.format(len(dist[ii]))
					else: lbl_=lbl[jj]
					c,b,i = axs[ii].hist(dist[ii],bins=nbins,facecolor=clr[jj],edgecolor='k',alpha=0.7,density=True,label=lbl_)
					if np.amin(dist[ii])<dist_ex[ii][0]: dist_ex[ii][0]=np.amin(dist[ii])
					if np.amax(dist[ii])>dist_ex[ii][1]: dist_ex[ii][1]=np.amax(dist[ii])
					if np.amax(c)>dist_ex[ii][2]: dist_ex[ii][2]=np.amax(c)
					if ii==2:
						print '###########################\n{}; dist {}'.format(lbl[jj],ii)
						print 'N={} ; Median={}; Avg={}; Std={}'.format( len(dist[ii]),np.median(dist[ii]),np.average(dist[ii]),np.std(dist[ii]) )
			print 'total =', total,'; missing =', missing
			if len(cat_cmp)!=0:
				print 'Including extra catalogue data'
				ind = np.unique(cat_cmp['sdss_name'],return_index=True)[1]
				cols=['logbol','logbh','logEdd','z']
				for ii in range(len(dist_ex)):
					dist = cat_cmp[cols[ii]][ind]
					c,b,i = axs[ii].hist(dist,bins=9,histtype='step',density=True,fill=False,edgecolor='k',lw=1.5,label=lbl[-1])
					if np.amin(dist)<dist_ex[ii][0]: dist_ex[ii][0]=np.amin(dist)
					if np.amax(dist)>dist_ex[ii][1]: dist_ex[ii][1]=np.amax(dist)
					if np.amax(c)>dist_ex[ii][2]: dist_ex[ii][2]=np.amax(c)
					print '###########################\n{}; dist {}'.format(lbl[-1],ii)
					print 'N={} ; Median={}; Avg={}; Std={}'.format( len(dist),np.median(dist),np.average(dist),np.std(dist) )
			for ii in range(len(dist_ex)):
				axs[ii].tick_params(labelleft=0,right=1,direction='in',labelsize=14,which='both')
				axs[ii].axis([dist_ex[ii][0],dist_ex[ii][1],0,1.05*dist_ex[ii][2]])
			axs[0].legend(loc='upper right',numpoints=1, edgecolor='k',fontsize=16)
			axs[2].set_ylabel(r'Normalised N$_{spec}$',fontsize=18,fontweight='bold')
			axs[0].set_xlabel(r'log$_{10}$ L$_{bol}$',fontsize=18,fontweight='bold')
			axs[1].set_xlabel(r'log$_{10}$ M$_{BH}$',fontsize=18,fontweight='bold')
			axs[2].set_xlabel(r'log$_{10}$ L$_{bol}$/L$_{Edd}$',fontsize=18,fontweight='bold')
			axs[3].set_xlabel(r'z',fontsize=14,fontweight='bold')
			plt.tight_layout()
		elif  fig_type=='flux_vs_dt' or fig_type=='continuum_vs_dt':
			if fig_type == 'continuum_vs_dt': tt=12; title_str = r'$\Delta$ 2798${\AA}$ Continuum Flux'
			else: tt=14; title_str = r'$\Delta$ MgII Line Flux'
			print fig_type
			fig = plt.figure(figsize=(7,6))
			gs = gridspec.GridSpec(3,3)
			gs.update(wspace=0,hspace=0)
			axs = [fig.add_subplot(gs[0:2,0:3]),fig.add_subplot(gs[2:3,0:1]),fig.add_subplot(gs[2:3,1:2]),fig.add_subplot(gs[2:3,2:3])]
			dt_arr = self.data[6]/(self.data[4]+1)
			blrng = mcl.LinearSegmentedColormap.from_list('blrng', colors=['#8DEEEE','#00008B'])
			counts,xb,yb,image = plt.hist2d(dt_arr,self.data[tt],bins=nbins,range=[[0,np.amax(dt_arr)],[lims[-2],lims[-1]]])
			plt.cla()
			axs[0].axis([0,np.amax(dt_arr),lims[-2],lims[-1]])
			extent = [xb[0],xb[-1],yb[0],yb[-1]]
			im = axs[0].imshow(counts.T,aspect='auto',extent=extent,cmap=blrng,origin='lower',norm=mcl.LogNorm(vmin=1,vmax=np.amax(counts)))
			cax = fig.add_axes([0.91, 0.4, 0.02, 0.4])
			cb = fig.colorbar(im, cax, orientation='vertical',extend='max')#, ticks=tks)
			cb.ax.tick_params(labelsize=15)
			axs[0].set_ylabel(title_str,fontsize=15,fontweight='bold')
			axs[0].set_xlabel(r'$\Delta$t in QSO restframe (days)',fontsize=15,fontweight='bold')
			axs[0].xaxis.set_label_position('top')
			axs[0].tick_params(direction='in',left=1,right=0,top=1,bottom=1,labelbottom=0,labeltop=1)
			plt.text(1, 1.03, r'$N_{spec}$', fontsize=13)
			if dt_rng:
				dt_rng = [(),dt_rng[0],dt_rng[1],dt_rng[2]]
			else:
				dt_rng = [(),(75,125),(1400,1600),(2500,3500)]
			box = dict(boxstyle='square',facecolor='w')
			for ii in [1,2,3]:
				dat = self.data[tt][np.argwhere((dt_arr>dt_rng[ii][0]) & (dt_arr<dt_rng[ii][1]) & (self.data[tt]>-7) & (self.data[tt]<7))].flatten()
				dat_cl = self.data[tt][np.argwhere((dt_arr>dt_rng[ii][0]) & (dt_arr<dt_rng[ii][1]) & (self.data[tt]>-2) & (self.data[tt]<2))].flatten()
				nbins = int(len(dat)/15)
				counts,bins,image = axs[ii].hist(dat_cl,nbins,cumulative=1,facecolor='r',alpha=0.4,zorder=10)
				frng = np.linspace(lims[-2],lims[-1],100)
				nrm =  np.amax(counts)
				axs[ii].plot(frng,nrm*stat.norm.cdf(frng,loc=np.average(dat),scale=np.std(dat)),'--',c='k')
				axs[ii].axis([-bins[-1],bins[-1],0,1.3*nrm])
				axs[ii].text(.05,.87,r'{}<$\Delta$t<{}'.format(dt_rng[ii][0],dt_rng[ii][1]),transform=axs[ii].transAxes,bbox=box,zorder=11,fontsize=13)
				axs[ii].tick_params(labelleft=0)#,labelbottom=0)
				axs[ii].set_yticks([.25*nrm,.5*nrm,.75*nrm,nrm])
				print '###########################'
				print 'dt =',dt_rng[ii],': Nspec =',len(dat)
				print 'mu = {}; sigma = {}; skew ={}; kurtosis = {}'.format(np.average(dat),np.std(dat),stat.skew(dat),stat.kurtosis(dat))
				print 'LF =',lilliefors(dat,dist='norm')

			axs[1].tick_params(labelleft=1,labelbottom=1)
			axs[1].set_yticklabels(['0.25','0.5','0.75','1'])
			axs[1].set_ylabel(r'Normalised $N_{spec}$',fontsize=13,fontweight='bold')
			axs[1].set_xlabel(title_str,fontsize=13,fontweight='bold')
		elif fig_type=='response_metric_hist':
			RM = []
			for ii in range(len(self.data[0])):
				rm = self.data[2][ii]-self.data[0][ii]
				if self.data[2][ii]==self.data[0][ii]: print self.data[2][ii] #check on data quality
				if abs(rm)<10:
					RM+=[rm/np.sqrt(2)]
			fig,ax = plt.subplots(1,figsize=(5,4))
			if len(additional_data)==0:
				ax.hist(RM,nbins,facecolor='seagreen',edgecolor='k')
				ax.set_ylabel(r'N$_{spec}$',fontweight='bold',fontsize=16)
			else:
				ad = additional_data; RMa=[]
				for ii in range(len(ad[0])):
					rm = ad[2][ii]-ad[0][ii]
					if self.data[2][ii]==self.data[0][ii]: print self.data[2][ii]
					if abs(rm)<10:
						RMa += [rm/np.sqrt(2)]
				ax.hist(RM,nbins,facecolor='seagreen',edgecolor='seagreen',density=True,alpha=0.75,label='Full Population')
				ax.hist(RMa,9,facecolor='None',edgecolor='k',density=True,label='Supervariable')
				ax.set_ylabel(r'N$_{spec}$ (Normalised)',fontweight='bold',fontsize=16)
				ax.legend(loc='upper right',edgecolor='k')
			ax.set_xlabel(r'($\Delta f_{2798} > \Delta f_{MgII}$)   $\alpha_{rm}$   ($\Delta f_{2798} < \Delta f_{MgII}$)',fontweight='bold',fontsize=16)
			if ylog==True:
				ax.set_yscale('log')
			plt.tight_layout()
		elif fig_type=='cmp_clq':
			try: clq_nl = np.loadtxt(clq_namelist,usecols=0,dtype=str)
			except: print 'Please provide a valid CLQ namelist'; return
			clq,no_clq=[],[]
			RM=[]
			for ii in range(len(self.namelist)):
				try: #catch end of the namlist
					if self.namelist[ii]!=self.namelist[ii+1]:
						RM+=[(self.data[2][ii]-self.data[0][ii])/np.sqrt(2)]
						rm_in=np.average(RM); RM=[]
						if self.namelist[ii] in clq_nl:
							clq+=[rm_in]
						else: no_clq+=[rm_in]
				except:
					RM+=[(self.data[2][ii]-self.data[0][ii])/np.sqrt(2)]
					rm_in=np.amax(RM)
					if self.namelist[ii] in clq_nl:
						clq+=[rm_in]
					else: no_clq+=[rm_in]
				RM+=[(self.data[2][ii]-self.data[0][ii])/np.sqrt(2)]
			fig,ax = plt.subplots(1,figsize=(4,3))
			print '     CLQ: N = {}, avg. = {}, median = {}'.format(len(clq),np.average(clq),np.median(clq))
			print 'non- CLQ: N = {} avg. = {}, median = {}'.format(len(no_clq),np.average(no_clq),np.median(no_clq))
			ax.hist([clq,no_clq],nbins,density=0,color=['dodgerblue','r'],edgecolor='k',alpha=0.75,label=['CLQ: N={}'.format(len(clq)),'Non-CLQ: N={}'.format(len(no_clq))] )
			ax.set_ylabel(r'N$_{spec}$',fontweight='bold',fontsize=14)
			leg = ax.legend(loc='upper left',edgecolor='k')
			ax.set_xlabel(r'$\alpha_{rm}$',fontweight='bold',fontsize=14)
			for t in leg.get_texts():
				t.set_ha('right') # ha is alias for horizontalalignment
				t.set_position((75,0))
			plt.tight_layout()
		elif fig_type=='del_sigma_hist':
			fig,ax=plt.subplots(1,figsize=(6,4))
			din = self.data[10]/(1+self.data[4])
			din = din[abs(din)<200]
			din = din[din!=0]
			ax.hist(din,nbins,facecolor='grey',edgecolor='k',density=1,label='N = 15,101')
			ax.set_xlabel(r'$\Delta\sigma_{\mathrm{MgII}}$ $({\rm\AA})$',fontsize=16,fontweight='bold')
			ax.set_ylabel(r'N$_{spec}$ (Normalised)',fontsize=16,fontweight='bold')
			ax.tick_params('x',labelsize=14)
			ax.tick_params('y',which='both',labelleft=0,length=0)
			if len(additional_data)!=0:
				ad = additional_data
				counts,bins,image = ax.hist(ad[10]/(1.+ad[4]),7,facecolor='y',alpha=0.7,zorder=9,edgecolor='k',density=1,label='N = 108')				
			if ylog==True:
				ax.set_yscale('log')
			ax.legend(loc='upper left',fontsize=14)
			plt.tight_layout()
		elif fig_type=='sig_vs_con' or fig_type=='delsig_vs_delcon' or fig_type=='sig_vs_line' or fig_type=='delsig_vs_delline':
			d_corr=1e14; c=299792458
			xu,yu = '$(10^{42}$ erg s$^{-1})$','$($km/s$)$'
			if fig_type=='sig_vs_con':
				xi=0; yi=8; xlbl=r'$2798{\rm\AA}$ Continuum Flux'; ylbl=r'$\sigma_{\mathrm{MgII}}$'
				if lum=='con': d_corr = 4*np.pi*np.power(cosmo.comoving_distance(self.data[4]).cgs.value*(1.+self.data[4]),2)*2798*1e-42; xlbl=r'$\lambda$L$_{2798}$'
			elif fig_type=='delsig_vs_delcon':
				xi=12; yi=10; xlbl=r'$\Delta$ $(2798{\rm\AA}$ Continuum Flux$)$'; ylbl=r'$|\Delta\sigma_{\mathrm{MgII}}$ $(\rm\AA)$|'
				if lum=='con': d_corr = 4*np.pi*np.power(cosmo.comoving_distance(self.data[4]).cgs.value*(1.+self.data[4]),2)*2798*1e-42; xlbl=r'$|\Delta\lambda \mathrm{L}_{2798}|$'
			elif fig_type=='sig_vs_line':
				xi=2; yi=8; xlbl=r'$f_{\mathrm{MgII}}$ ($10^{14}$ erg cm$^{-2}$ s$^{-1}$)'; ylbl=r'$\sigma_{\mathrm{MgII}}$'
				if lum=='line': d_corr = 4*np.pi*np.power(cosmo.comoving_distance(self.data[4]).cgs.value*(1.+self.data[4]),2)*1e-42; xlbl=r'L$_{\mathrm{MgII}}$'
			elif fig_type=='delsig_vs_delline':
				xi=14; yi=10; xlbl=r'$\Delta$ $($MgII Line Flux$)$'; ylbl=r'$\Delta\sigma_{\mathrm{MgII}}$'
				if lum=='line': d_corr = 4*np.pi*np.power(cosmo.comoving_distance(self.data[4]).cgs.value*(1.+self.data[4]),2)*1e-42; xlbl=r'|$\Delta$L$_{\mathrm{MgII}}$| $(10^{42}$ erg s$^{-1})$'
			print 'making', fig_type
			if fig_type[:3]!='del':
				X,Y = self.data[xi]*d_corr,self.data[yi]/(1+self.data[4])*(c/2798)*1e-3
				Xerr,Yerr = self.data[xi+1]*d_corr,self.data[yi+1]/(1+self.data[4])*(c/2798)*1e-3
			else:
				X,Y=self.data[xi],self.data[yi]
				Xerr,Yerr=self.data[xi+1],self.data[yi+1]
			X = X[Y!=0]; Xerr=Xerr[Y!=0]; Yerr=Yerr[Y!=0];Y=Y[Y!=0]
			if contours==True:
				fig,ax=plt.subplots(1,figsize=(7.5,6))
				if fig_type[:3]!='del':
					if xlog==1: xdat = np.log10(self.data[xi]*d_corr); lims[0]=np.log10(lims[0]); lims[1]=np.log10(lims[1]); xlbl=r'log$_{10}($'+xlbl+'$)$ '+xu
					else: xdat=self.data[xi]*d_corr
					if ylog==1: ydat=np.log10(self.data[yi]/(1+self.data[4])*(c/2798)*1e-3); lims[2]=np.log10(lims[2]); lims[3]=np.log10(lims[3]); ylbl=r'log$_{10}($'+ylbl+'$)$ '+yu
					else: ydat=self.data[yi]/(1+self.data[4])*(c/2798)*1e-3
				else:
					xdat,ydat=self.data[xi],self.data[yi]
				counts,xb,yb,image = plt.hist2d(xdat,ydat,bins=nbins,range=[lims[:2],lims[2:]])
				plt.cla()
				m = np.amax(counts)
				levels_pct = np.array(levels_pct)*m
				extent = [xb[0],xb[-1],yb[0],yb[-1]]
				if log_counts==True:
					norm = cm.colors.LogNorm()
				else: norm=None
				im = ax.imshow(counts.T,extent=extent,cmap=cmap,aspect='auto',origin='lower',norm=norm)
				ax.contour(counts.T,extent=extent,levels=levels_pct)
				ax.axis(lims)
				if len(additional_data)!=0:
					ad=additional_data
					markers, caps, bars = ax.errorbar(ad[xi],ad[yi],xerr=ad[xi+1],yerr=ad[yi+1],fmt='o',markersize=7,c='red',markeredgecolor='None',ecolor='red',capsize=1.5)
					a = stat.linregress(ad[xi],ad[yi]); print a
					ax.plot(np.linspace(lims[0],lims[1],10),a[0]*np.linspace(lims[0],lims[1],10)+a[1],c='r',linestyle='--',zorder=100)
					[bar.set_alpha(0.5) for bar in bars]
					[cap.set_alpha(0.5) for cap in caps]
				cax = fig.add_axes([0.83, 0.2, 0.02, 0.7])
				cax.text(0.1,1.02,'Counts',transform=cax.transAxes,fontweight='bold',fontsize=13)
				fig.colorbar(im,cax,orientation='vertical')
			else:
				fig,ax=plt.subplots(1,figsize=(6,5))
				if absval==0:
					ax.errorbar(X,Y,xerr=Xerr,yerr=Yerr,fmt='o',markersize=7,c='grey',markeredgecolor='k',ecolor='k',capsize=2)
				else:
					ax.scatter(np.absolute(X),np.absolute(Y),marker='o')
					ax.errorbar(np.absolute(X),np.absolute(Y),xerr=Xerr,yerr=Yerr,fmt='o',markersize=7,c='grey',markeredgecolor='k',ecolor='k',capsize=2)
				a = stat.linregress(X,Y); print a
				ax.plot(np.linspace(-10,10,10),a[0]*np.linspace(-10,10,10)+a[1],c='r',linestyle=':',zorder=0)
				ax.axis(lims)
				if ylog==1: ax.set_yscale('log')
				if xlog==1: ax.set_xscale('log')
			ax.set_xlabel(xlbl,fontsize=18,fontweight='bold')
			ax.set_ylabel(ylbl,fontsize=18,fontweight='bold')
			ax.tick_params(axis='both',labelsize=15)
			ax.minorticks_on()
			plt.tight_layout()
		elif fig_type=='cvsf_individual':
			fig, ax = plt.subplots(1,figsize=(6,5),dpi=150)
			dmin,dmax=52100,57600
			dates = [52200,52944,55179,55208,55445,55476,55827,55856,55945,56219,56544,56568,56596]
			tks = np.arange(dmin,dmax,1500)
			blrng = mcl.LinearSegmentedColormap.from_list('blrng', colors=['#8DEEEE','#000055'])
			im = ax.scatter(self.data[0], self.data[2], c=dates, s=160, cmap=blrng, vmin=dmin, vmax=dmax, zorder=2,edgecolor='k')
			ax.errorbar(self.data[0], self.data[2], xerr=self.data[1], yerr=self.data[3], fmt='None', ecolor='k', elinewidth=.8, zorder=1,capsize=3)
			ax.plot(np.linspace(0,2,10),np.linspace(0,2,10),linestyle='--',color='grey',zorder=0)
			cax = fig.add_axes([0.78, 0.18, 0.02, 0.4])
			cb = fig.colorbar(im, cax, orientation='vertical', ticks=tks)
			cb.ax.tick_params(labelsize=16)
			plt.text(1.9, 1.02, 'MJD', fontsize=14,fontweight='bold')
			ax.set_ylim([0,1.09])
			ax.set_xlim([0,1.09])
			ax.set_xlabel(r'2798$\rm\AA$ Continuum Flux', fontsize=20, fontweight='bold')
			ax.set_ylabel('MgII Line Flux', fontsize=20, fontweight='bold')
			ax.minorticks_on()
			fig.tight_layout()
		elif fig_type=='ew_histogram':
			fig, ax = plt.subplots(1,figsize=(6,4),dpi=150)
			d_corr = 4*np.pi*np.power(cosmo.comoving_distance(self.data[6]).cgs.value*(1.+self.data[6]),2)*2798
			ew = np.log10(self.data[4]/(1+self.data[6]))
			c = np.log10(self.data[0]*d_corr)
			iBE = []; lim=7
			for ii in reversed(range(1,len(self.namelist))):
				if self.namelist[ii]==self.namelist[ii-1]:
					ibe = (ew[ii]-ew[ii-1])/(c[ii]-c[ii-1])
					if abs(ibe)<lim:
						iBE+=[ibe]
			ibe=(ew[1]-ew[0])/(c[1]-c[0])
			if abs(ibe)<lim: iBE+=[ibe]
			ax.hist(iBE,nbins,facecolor='b',edgecolor='k',alpha=0.5)
			ax.set_xlabel('intrinsic Baldwin Effect',fontsize=17,fontweight='bold')
			ax.set_ylabel(r'N',fontsize=17,fontweight='bold')
			box = dict(boxstyle='square',facecolor='w')
			ax.text(0.7,0.8,'N = {}\n'.format(len(iBE))+r'$\mu$ = {}'.format(round(np.average(iBE),2)),transform=ax.transAxes,bbox=box,fontsize=16)
			ax.set_xlim(-lim,lim)
			print 'N = {}, avg = {}, median = {}'.format(len(iBE),np.average(iBE),np.median(iBE))
			plt.tight_layout()
		elif fig_type=='lzedd':
			try:
				col_names = ('spec_name','sdss_name','z','logl3000','logbol','logbh','logEdd','rloud','ewmgii')
				formats = ('S30','S30')+7*('float',)
				self.catalogue = np.loadtxt(catalogue_file,dtype={'names':col_names,'formats':formats})
				if cat_cmp:
					cat_cmp = np.loadtxt(cat_cmp,dtype={'names':col_names,'formats':formats}).T
			except: print 'Please create catalogue_file using make_catalogue_file'; return
			names=[self.namelist]
			#Create the distributions
			#-1- Full Population
			total, missing = 0,0
			dist_ex=[[100,0,0],[100,0,0],[100,0,0],[100,0,0]] #Used to control the axis in the plot
			for jj in range(len(names)):
				dist=[[],[],[],[]]
				for obj in names[jj]:
					total+=1
					ind = np.where(self.catalogue['spec_name']==obj)
					if len(ind[0])!=0:
						dist[3] += [ self.catalogue['z'][ind][0] ]
						dist[2] += [ self.catalogue['logEdd'][ind][0] ]
						dist[1] += [ self.catalogue['logbh'][ind][0] ]
						dist[0] += [ self.catalogue['logbol'][ind][0] ]	
					else: missing+=1
			#-2- Supervariable
			ind = np.unique(cat_cmp['sdss_name'],return_index=True)[1]
			cols=['logbol','logbh','logEdd','z']
			dist_sv = [[],[],[],[]]	
			for ii in range(len(dist_ex)):
				dist_sv[ii] = cat_cmp[cols[ii]][ind]
			#Create the figure
			fig,ax = plt.subplots(1,figsize=(5.5,4))
			lbl=['Full Population','Supervariable']
			hp = ax.scatter(dist[3],dist[0],marker='o',c=dist[2],norm=cm.colors.Normalize(-3,1),cmap=cmap,label=lbl[0],s=10)
			ax.scatter(dist_sv[3],dist_sv[0],marker='s',edgecolors='k',c=dist_sv[2],norm=cm.colors.Normalize(-3,1),cmap=cmap,label=lbl[1])
			ax.axis(lims)
			ax.set_xlabel('$z$',fontsize=18,fontweight='bold')
			ax.set_ylabel(r'log$_{10}$L$_{\rm bol}$',fontsize=18,fontweight='bold')
			cax = fig.colorbar(hp, ax=ax, extend='max')
			cax.set_label(r'log$_{10}$ $\lambda_{\rm Edd}$')
			ax.legend(loc='upper left',edgecolor='k',fontsize=14)
			plt.tight_layout()
		else:
			print 'Invalid fig_type selected'
			return
		plt.savefig(saveloc+name)
		plt.close()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#









