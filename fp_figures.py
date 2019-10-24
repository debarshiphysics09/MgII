import sys
sys.path.append('') #Append the location of Norm_and_Fit.py if necessary
from Norm_and_Fit import Normalisation, Fit
import numpy as np

##Normalisation: Full SDSS sample (no norm,min, max and EW)
#sdss = Normalisation(namelist='mgii_sample_ext_list',exclude='sdss_failed',deltat_limit=30)
#sdss.load_data(data_path='./data/Fe_T_V/',inc_double_sig=True)
#sdss.create_flux_list(name_add='_sdss')
#del sdss
#sdss = Normalisation(namelist='mgii_sample_ext_list',exclude='sdss_failed',deltat_limit=30)
#sdss.load_data(data_path='./data/Fe_T_V/',inc_double_sig=True)
#sdss.normalise('cmax')
#sdss.create_flux_list(name_add='_sdss_max')
#del sdss
#sdss = Normalisation(namelist='mgii_sample_ext_list',exclude='sdss_failed',deltat_limit=30)
#sdss.load_data(data_path='./data/Fe_T_V/',inc_double_sig=False)
#sdss.normalise('cmin')
#sdss.create_flux_list(name_add='_sdss_min')
#del sdss
#sdss = Normalisation(namelist='mgii_sample_ext_list',exclude='sdss_failed',deltat_limit=30)
#sdss.load_data_ew(data_path='./data/Fe_T_V/')
#sdss.create_ew_list(name_add='_sdss')
#del sdss

###################################
##Figures


##Delta t histogram (Figure 1)
#F = Fit()
#F.load_data('MgII_norm_flux_var_max')
#F.filter_data()
#var_data = np.copy(F)
#del F; F = Fit()
#F.load_data('MgII_norm_flux_sdss_max')
#F.filter_data()
#F.create_figure(saveloc='./plots/',fig_type='dt_histogram',additional_data=var_data,name='dt_histogram_both.pdf',lims=[0,1.,0,2.],nbins=20,ylog=True)
#del F,var_data

##Preliminary tests; these are used to calculate some of the data in Tables 6 & 7
#F = Fit()
#F.load_data('MgII_norm_flux_sdss_max')
#F.filter_data(norm_clean_max=True)
#print 'Normalised to maximum continuum:'
#F.find_spearman(prnt=True)
#F.find_pearson_pval(prnt=True)
##F.bootstrap_mc_cmp(Nmc=1000)
#mod_cmp=F.compare_models(prnt=True)
#print mod_cmp['fname']
#print mod_cmp['parameters']
#print mod_cmp['red_chi2']
#print mod_cmp['seq']
#del F
#print '##############################'
#F = Fit()
#F.load_data('MgII_norm_flux_sdss_min')
#F.filter_data(norm_clean_min=True)
#print 'Normalised to minimum continuum:'
#F.find_spearman(prnt=True)
#F.find_pearson_pval(prnt=True)
##F.bootstrap_mc_cmp(Nmc=1000)
#mod_cmp=F.compare_models()
#print mod_cmp['fname']
#print mod_cmp['parameters']
#print mod_cmp['red_chi2']
#print mod_cmp['seq']
#del F
#print '##############################'

#2D histogram contours (Figures 5 & 6)
#F = Fit()
#F.load_data('MgII_norm_flux_sdss_max')
#F.filter_data(norm_clean_max=True)
#F.create_figure(saveloc='./plots/',fig_type='colourmap',name='sdss_2dhist_max.pdf',lims=[0,1.,0,1.7],nbins=25,levels_pct=[.1,.25,.5,.75,.9],log_counts=1)
#del F
#F = Fit()
#F.load_data('MgII_norm_flux_sdss_min')
#F.filter_data(norm_clean_min=True)
#F.create_figure(saveloc='./plots/',fig_type='colourmap',name='sdss_2dhist_min.pdf',lims=[1,1.7,0,1.7],nbins=20,levels_pct=[.1,.25,.5,.75,.9],log_counts=1)
#F.create_figure(saveloc='./plots/',fig_type='colourmap',name='sdss_2dhist_min.pdf',lims=[1,8,0,4],nbins=50,levels_pct=[.1,.25,.5,.75,.9],log_counts=True,cmap='inferno')
#del F

##2D-histogram divided into Delta t panels (Figure 12)
#F = Fit()
#F.load_data('MgII_norm_flux_sdss_min')
#F.filter_data(norm_clean_min=True)
#F.create_figure(saveloc='./plots/',fig_type='dtbin_colourmap',name='sdss_dtbin.pdf',lims=[1,1.7,0,1.7],nbins=15,dtbins=[1,2,3.5,10],levels_pct=[.2,.35,.5,.75,.9])
#del F

##Responsivity Measure histogram (Figure 7)
#F = Fit()
#F.load_data('MgII_norm_flux_var_min')
#F.filter_data()
#var_data = np.copy(F.data)
#del F; F= Fit()
#F.create_figure(saveloc='./plots/',fig_type='response_metric_hist',additional_data=var_data,name='resp_histogram_sdss.pdf',nbins=15,ylog=True)
#del F,var_data

##flux versus Delta t (Figure 13)
#F = Fit()
#F.load_data('MgII_norm_flux_sdss_max')
#F.filter_data(delta=1,norm_clean_max=False)
#F.create_figure(saveloc='./plots/',fig_type='flux_vs_dt',name='ldt_sdss.pdf',lims=[0,0,-2.,2.],dt_rng=[(0,500),(1000,1500),(2000,3500)],nbins=40)
#del F

##Histogram of Delta sigma (Figure 14)
#Fvar=Fit()
#Fvar.load_data('/disk1/dhoman/Project/Python/Line_Fit/WHT_Fit/Plot_CLQ/MgII_flux_var')
#Fvar.filter_data(delta=True)
#F=Fit()
#F.load_data('MgII_flux_sdss')
#F.filter_data(delta=True)
#F.create_figure(saveloc='./plots/',fig_type='del_sigma_hist',name='del_sig_histogram_both.pdf',nbins=19,ylog=1,additional_data=Fvar.data)
#del F,Fvar

#2D histograms of sigma & Delta_sigma
##a) Sigma vs line flux (Figure 16)
#F=Fit()
#F.load_data('MgII_flux_sdss')
#F.filter_data()
#F.create_figure(saveloc='./plots/',fig_type='sig_vs_line',name='sig_vs_line.pdf',lims=[1e1,1e3,10**3.,10**3.7],contours=True,nbins=20,levels_pct=[.4,.6,.75,.9],xlog=1,ylog=1,lum='line',cmap='Greys')
#del F
#b) #Combination of var and full population (Figure 15)
#Fvar=Fit()
#Fvar.load_data('/disk1/dhoman/Project/Python/Line_Fit/WHT_Fit/Plot_CLQ/MgII_norm_flux_var_max') #Use the normalisation to the oldest epoch for del_sig and del_line
#Fvar.filter_data(delta=True,rm_delsig=True)
#F=Fit()
#F.load_data('MgII_norm_flux_sdss_max')
#F.filter_data(delta=True,rm_delsig=True)
#F.create_figure(saveloc='./plots/',fig_type='delsig_vs_delline',name='delsig_vs_delline_both.pdf',contours=True,nbins=25,levels_pct=[.05,.15,.25,.5,.75],lims=[-1.,1.1,-1.,1.],cmap='Greys',log_counts=1,additional_data=Fvar.data)
#del F,Fvar

##Histograms of values from Shen et al. (2011) catalogue combined with plot showing the regions on the 2D histogram (Figure 20)
#a) Make catalogue file; this file is used to create other plots as well (see below)
#F = Fit()
#F.load_data('MgII_norm_flux_sdss_min',names='Named_MgII_norm_flux_sdss_min')
#F.make_catalogue_file('/disk1/dhoman/Project/Data/dr7q_Shen.fits',outfile='catalogue_data_sdss',use_sdss_name=True)
# del F
#b) Create plot
#F=Fit()
#F.load_data('MgII_norm_flux_sdss_max',names='Named_MgII_norm_flux_sdss_max')
#F.filter_data(norm_clean_max=1)
#nbins=25; boxs=[((.1,0.1),(0.45,0.45)),((0.75,0.85),(1.,1.05)),((0.75,0.1),(1.,0.45))]
#F.create_figure(saveloc='./plots/',fig_type='subsample_catalogue',name='catS7_overview_max.pdf',catalogue_file='catalogue_data_sdss',lims=[0,1,0,1.7],nbins=nbins,boxs=boxs,log_counts=True,nbins_sub=9)
#del F

##Histograms of DR7 values for different RM subsamples (Figure 21)
#F = Fit()
#F.load_data('MgII_norm_flux_sdss_min',names='Named_MgII_norm_flux_sdss_min')
#F.filter_data(name_pair=True)
#cat_cmp = '/disk1/dhoman/Project/Python/Line_Fit/WHT_Fit/Plot_CLQ/catalogue_data_var'
#F.create_figure(saveloc='./plots/',fig_type='response_catalogue',name='resp_cats7.pdf',catalogue_file='catalogue_data_sdss',cat_cmp='',rm_cat=1,nbins=9,rm_lims=((-100,-.75),(.75,100)) )
#del F

##Intrinsic Baldwin Effect histogram (Figure 11)
#F=Fit()
#F.load_data('MgII_EW_sdss',names='Named_MgII_EW_sdss')
#F.filter_data()
#F.create_figure(saveloc='./plots/',fig_type='ew_histogram',name='iBE_hist_sdss.pdf',nbins=21)
#del F

#Linear regression delta L on delta C (used for comparison with Yang et al. (2019) )
#F=Fit()
#F.load_data('MgII_flux_sdss')
#F.filter_data()
#F.linregress_lvsc(val='con_sig',lim=10)
#del F

#Compare Lbol, z, and Eddington ratio (Figure 19)
#F = Fit()
#F.load_data('MgII_norm_flux_sdss_min',names='Named_MgII_norm_flux_sdss_min')
#cat_cmp = '/disk1/dhoman/Project/Python/Line_Fit/WHT_Fit/Plot_CLQ/catalogue_data_var'
#F.filter_data(name_pair=True)
#F.create_figure(saveloc='./plots/',fig_type='lzedd',name='lbolzedd.pdf',catalogue_file='catalogue_data_sdss',cat_cmp=cat_cmp,lims=[0.2,2.3,44.5,48.5],cmap='coolwarm' )
#del F












