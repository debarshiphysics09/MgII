import sys
sys.path.append('') #Append the location of Norm_and_Fit.py if necessary
from astropy.io import fits
import numpy as np
from Norm_and_Fit import Normalisation,Fit

#namelist = '' #location of a file that lists all the spectra (in ASCII format)
#sp_nl = np.loadtxt('Namelist_MgII_var_expl',dtype='str').T[0]
#np.savetxt('Namelist_MgII_var',sp_nl,fmt='%s')
#spec_namelist = 'Namelist_MgII_var'

################################################

##Normalisation (no norm,min, max and EW)
#norm = Normalisation(namelist=namelist,spec_namelist=spec_namelist)
#norm.load_data(data_path='../profileFitterRaw/data/Parallel_Fit/',inc_double_sig=True)
#norm.load_mc_errors(data_path='../profileFitterRaw/data/MC/',nmc=250)
#norm.create_flux_list(name_add='_var')
#del norm
#norm = Normalisation(namelist=namelist,spec_namelist=spec_namelist)
#norm.load_data(data_path='../profileFitterRaw/data/Parallel_Fit/',inc_double_sig=True)
#norm.load_mc_errors(data_path='../profileFitterRaw/data/MC/',nmc=250)
#norm.normalise('cmax')
#norm.create_flux_list(name_add='_var_max')
#del norm
#norm = Normalisation(namelist=namelist,spec_namelist=spec_namelist)
#norm.load_data(data_path='../profileFitterRaw/data/Parallel_Fit/',inc_double_sig=True)
#norm.load_mc_errors(data_path='../profileFitterRaw/data/MC/',nmc=250)
#norm.normalise('cmin')
#norm.create_flux_list(name_add='_var_min')
#del norm
#norm = Normalisation(namelist=namelist,spec_namelist=spec_namelist)
#norm.load_data_ew(data_path='../profileFitterRaw/data/Parallel_Fit/')
#norm.load_mc_errors(data_path='../profileFitterRaw/data/MC/',nmc=250,ew=True)
#norm.create_ew_list(name_add='_var')
#del norm


#################################################
##Figures

##One responsivity functions (and test statistics); normalisation=max (Figure 5)
#fit = Fit()
#fit.load_data('MgII_norm_flux_var_max')
#fit.filter_data()
#print '##############################'
#print 'Normalised to maximum continuum:'
#fit.find_spearman(prnt=True)
#fit.find_pearson_pval(prnt=True)
##fit.bootstrap_mc_cmp(Nmc=10000)
#mod_cmp=fit.compare_models()
##fit.bootstrap_mc_cmp(fname=['fit_line'],Nmc=10000)
#print 'making figure'
#mod_cmp=fit.compare_models(fname=['fit_line'],prnt=False)
#print mod_cmp['parameters']
#fit.create_figure(name='mgii_linfit_var_max.pdf',lims=[0,1.05,0,1.45],mod_cmp=mod_cmp)
#del fit

##Two responsivity functions (and test statistics); normalisation=min (Figure 6)
#fit = Fit()
#fit.load_data('MgII_norm_flux_var_min')
#fit.filter_data()
#print '##############################'
#print 'Normalised to minimum continuum:'
#fit.find_spearman(prnt=True)
#fit.find_pearson_pval(prnt=True)
##fit.bootstrap_mc_cmp(fname=['fit_line'],Nmc=10)
##mod_cmp=fit.compare_models(fname=['fit_line'],prnt=False)
##fit.bootstrap_mc_cmp(Nmc=10000)
#mod_cmp=fit.compare_models(prnt=True)
#print mod_cmp['parameters']
#fit.create_figure(name='mgii_linfit_var_min.pdf',lims=[0,12,0,6],mod_cmp=mod_cmp)
#del fit

#Responsivity of J022556 (Figure 10)
#fit=Fit()
#fit.load_data('MgII_norm_flux_var_max_022256')
#fit.create_figure(fig_type='cvsf_individual',name='CvsF_J022556.pdf')

##Overview of luminosities (Figure 4)
#fit=Fit()
#fit.load_data('MgII_flux_var')
#fit.filter_data(delta=True)
#fit.create_figure(fig_type='lum_overview',name='Luminosity_Change_var.pdf')
#del fit

#Make catalogue file based on Shen et al. (2011) DR7Q values (not used for figure in the case of the Supervariable Sample)
#F = Fit()
#F.load_data('MgII_norm_flux_var_min',names='Namelist_MgII_var_expl')
#F.make_catalogue_file('/disk1/dhoman/Project/Data/dr7q_Shen.fits',outfile='catalogue_data_var')
#del F

#Compare RM for CLQs and non-CLQs (Figure 8)
#F=Fit()
#F.load_data('MgII_norm_flux_var_min',names='Namelist_MgII_var_expl')
#F.filter_data()
#F.create_figure(fig_type='cmp_clq',name='rm_clq.pdf',clq_namelist='Namelist_CLQ',nbins=13)
#del F

#Linear regression delta L on delta C (used for comparison with Yang et al. (2019) )
#F=Fit()
#F.load_data('MgII_flux_var',names='Named_MgII_flux_var')
#F.filter_data()
#F.linregress_lvsc(sample='svs',val='con_sig')














