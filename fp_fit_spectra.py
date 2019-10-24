import sys
sys.path.append('') #Append the location of Spectrum.py if necessary
import numpy as np
import multiprocessing as mp
from Spectrum import Spectrum, SpectrumError
import time, datetime

path = '' #location of the lists created with fp.define
Sp_path = '/disk1/dhoman/Project/Data/SpectraDL/Spectra_DR14/'
failed_runs = 'failed_sdss_fits'
Fe_template='Vestergaard.txt'; dir_name = 'Fe_T_V'

file_list = np.loadtxt('mgii_sample_ext_list',dtype={'names':('name','obj','ra','dec','z','delta_t','airmass','S/N','A_v','A_v_sdss'),'formats':('S30','S20','float','float','float','float','float','float','float','float')})
#These are the fitting-routine iterations (as defined in the table in the accompanying document)
n_it_c = 2
n_it_cf = 4
n_it_l = 3
ncpu = mp.cpu_count()

def fit_spectra(file_list_in):
	#print bla
	for spec in file_list_in:

		spectrum = Spectrum(spec['name'],path=path,Spath=Sp_path,failed_runs=failed_runs,Fe_template=Fe_template)
		spectrum.load(); spectrum.z = spec['z']; spectrum.A_v = spec['A_v']; spectrum.obj = spec['obj']
		for s in ['+','-']:
			if len(spectrum.obj.split(s))!=1:
				s_tmp = ''
				for sl in spectrum.obj.split(s):
					s_tmp+=sl+'_'
				spectrum.obj=s_tmp[:-1]
				break
		spectrum.deredden()

		print 'SDSS spectrum: {}; ID: {}'.format(spectrum.full_name,spectrum.obj)

		#Fit the continuum
		try: 
			spectrum.trim_to_range(continuum='mgii')
		except SpectrumError: del spectrum; continue
		spectrum.set_models_iter(model='continuum_only')
		try:
			for ii in range(n_it_c):
				spectrum.run_fit()
				if ii != n_it_c-1:
					spectrum.use_fit_results(sigma_clip=2)
		except SpectrumError: del spectrum; continue
		spectrum.create_figure(dir_name=dir_name,fit='continuum')

		#Fit continuum and feii
		spectrum.set_models_iter(model='continuum_feii_only')
		try:
			ii=0
			while ii<n_it_cf:
				spectrum.run_fit()
				if ii != n_it_cf-1:
					spectrum.use_fit_results(sigma_clip=2)
				elif ii==n_it_cf-1 and spectrum.fit_out.params['FeUVamp'].value<0:
					ii=ii-1
					spectrum.use_fit_results(sigma_clip=2,limits={'FeUVamp':(0,np.inf),'powUVexp':(-np.inf,2)})
				ii+=1
		except SpectrumError: del spectrum; continue
		spectrum.create_figure(dir_name=dir_name,fit='continuum_feii')

		#Fit mgii
		try:
			spectrum.choose_model('MgII','MgII_2',n_it_l,sc=3)
		except SpectrumError: del spectrum; continue
		spectrum.create_figure(dir_name=dir_name,fit='mgii')

		spectrum.output_text(dir_name=dir_name)
		spectrum.create_figure(dir_name=dir_name,fit='combined')

		del spectrum

#Setup for parallel runs
sp_i = len(file_list)/ncpu
inp_split = []
for ii in range(1,ncpu):
	inp_split += [file_list[(ii-1)*sp_i:ii*sp_i]]
inp_split += [file_list[(ncpu-1)*sp_i:]]

#Main run
t_b = datetime.datetime.now()

p = mp.Pool(processes=ncpu)
p.map(fit_spectra,inp_split)

print 'Finished in: ',str(datetime.datetime.now()-t_b)






