import multiprocessing as mp
from astropy.io import fits
import numpy as np
import time, datetime, sys
from functools import partial
from astropy.io import fits

path = '' #path to storage location
Cat = path+'DR14Q_v4_4.fits' #filename of SDSS quasar catalogue DR14 (DR14Q)
Cat7 = path+'dr7q_Schneider.fits' #filename of SDSS quasar catalogue DR7 (DR7Q)
Cat7S = path+'dr7q_Shen.fits' #filename of quasar values catalogue based on DR7Q
path_SP = path+'SpectraDL/Spectra_DR14/' #Storage location of all SDSS spectra associated with quasars in DR14Q

hdu = fits.open(Cat); T14=hdu[1].data; hdu.close()
hdu = fits.open(Cat7S); T7S = hdu[1].data; hdu.close()
ncpu = mp.cpu_count()

##################################

def split_input(inp):
	'''Split an input array into a list of Ncpu arrays, where Ncpu is the number of available cores'''
	sp_i = len(inp)/ncpu
	inp_split = []
	for ii in range(1,ncpu):
		inp_split += [inp[(ii-1)*sp_i:ii*sp_i]]
	inp_split += [inp[(ncpu-1)*sp_i:]]
	print 'Dividing {} elements over {} CPU(s):'.format(len(inp),ncpu)
	for ii in range(ncpu):
		print 'inp_split bin',ii,len(inp_split[ii])
	return inp_split

def calc_extinction(inp):
	'''calculate extinction'''
	E_BV=[]
	SF_ugriz_values = [4.239,3.303,2.285,1.698,1.263]
	for ii in range(len(inp)):
		E_BV += [inp[ii]/SF_ugriz_values[ii]]
	avg = np.average(E_BV)
	return 2.742*avg

def if_mult_obs(mg14):
	'''Find spectral pairs for each object in the sample'''
	tmp=[]
	n_mult,n_it = 0,0
	n_to,n_to_add,n_mo,n_mo_add,n_calc,n_best,n_other = 0,0,0,0,0,0,0
	for obj in mg14:
		if n_it%10000==0: print mp.current_process(),n_it,n_mult,len(tmp)
		n_it+=1
		if obj['n_spec']==1:
			n_mult+=1
			n_to+=1
			MJD_best = obj['mjd']
			for ii in range(len(obj['mjd_duplicate'])):
				if obj['mjd_duplicate'][ii]!=-1 and obj['mjd_duplicate'][ii]!=0 and abs(MJD_best-obj['mjd_duplicate'][ii])>30:
					n_to_add+=1
					ext = calc_extinction(obj['gal_ext'])
					tmp+=[(obj['sdss_name'],obj['plate'],MJD_best,obj['fiberid'],obj['plate_duplicate'][ii],obj['mjd_duplicate'][ii],obj['fiberid_duplicate'][ii],obj['ra'],obj['dec'],obj['z'],obj['bi_civ'],ext)]
					break
		elif obj['n_spec']>1:
			n_mult+=1
			n_mo+=1
			MJD_best = obj['mjd']; mjd = [MJD_best]
			for ii in range(1,len(obj['mjd_duplicate'])):
				if obj['mjd_duplicate'][ii]!=0 and obj['mjd_duplicate'][ii]!=-1:
					mjd += [ obj['mjd_duplicate'][ii] ]
			if len(mjd)>1:
				n_calc+=1
				dif = []
				for ii in range(0,len(mjd)):
					for jj in range(ii+1,len(mjd)):
						dif += [[ ii,jj,abs(mjd[ii]-mjd[jj]) ]]
				dif = np.array(dif)
				dmax = np.amax(dif.T[2])
				if dmax>30:
					n_mo_add+=1
					for ii in range(len(dif)):
						if dif[ii][2] == dmax:
							cmp_id = np.array([dif[ii][0],dif[ii][1]])
							np.sort(cmp_id)
							break
					if cmp_id[0] == 0: #The best spectrum is part of the comparison
						n_best +=1
						cmp_id = 2*cmp_id-1 #Adjust to match the table formatting
						ext = calc_extinction(obj['gal_ext'])
						tmp+=[(obj['sdss_name'],obj['plate'],MJD_best,obj['fiberid'],obj['plate_duplicate'][cmp_id[1]],obj['mjd_duplicate'][cmp_id[1]],obj['fiberid_duplicate'][cmp_id[1]],obj['ra'],obj['dec'],obj['z'],obj['bi_civ'],ext)]
					else: #The comparison is made with two of the additional spectra
						n_other+=1
						cmp_id = 2*cmp_id-1
						ext = calc_extinction(obj['gal_ext'])
						tmp+=[(obj['sdss_name'],obj['plate_duplicate'][cmp_id[0]],obj['mjd_duplicate'][cmp_id[0]],obj['fiberid_duplicate'][cmp_id[0]],obj['plate_duplicate'][cmp_id[1]],obj['mjd_duplicate'][cmp_id[1]],obj['fiberid_duplicate'][cmp_id[1]],obj['ra'],obj['dec'],obj['z'],obj['bi_civ'],ext)]
	print 'finished', mp.current_process(),n_mult,'total =',len(tmp)
	return tmp

def remove_balq(cat,mgii):
	'''Remove BALs from the sample'''
	n_it,n_bal,n_bal7=0,0,0
	B = []
	for obj in cat:
		if obj['bal_flag']!=0: B+=[(obj[0],obj['ra'],obj['dec'])]
	print len(B)
	jstart=0;len_or=len(mgii)/2
	for ii in reversed(range(len(mgii)/2-1)):
		if mgii[ii*2]['bal_flag']!=0:
			n_bal+=1
			mgii=np.delete(mgii,(ii*2,ii*2+1))
		else: 
			for jj in range(jstart,len(B)):
				if mgii[ii]['sdss_name']==B[jj]:
					if round(mgii[ii*2]['ra'],5)==round(B[jj][2],5) and round(mgii[ii*2]['dec'],5)==round(B[jj][2],5):
						n_bal+=1
						n_bal7+=1
						mgii = np.delete(mgii,(ii*2,ii*2+1))
						jstart=jj
						break
		n_it+=1
		if n_it%5000==0: print 'Removing BALQs from sample of {}; n_it = {}; n_removed = {}; n_bal7 = {}'.format(len_or,n_it,n_bal,n_bal7)
	print 'Finished removing BALQs from sample of {}; n_it = {}; n_removed = {}; n_bal7 = {}'.format(len_or,n_it,n_bal,n_bal7)
	return mgii

def filter_z(path_SP,infile):
	'''Reduce spectrum based on z-range'''
	spec_list,coords=[],[]
	missing,z_issue,exposure,coord_mismatch=[],[],[],[]
	n_it = 0
	for jj in range(0,len(infile),2):

		n_it+=1
		if n_it%1000==0: print mp.current_process(),n_it,len(spec_list)

		names = [infile[jj]['spec_name'],infile[jj+1]['spec_name']]
		try:
			hdulist = fits.open(path_SP+names[0]); 
		except IOError:
			missing += [(infile[jj]['sdss_name'],names[0])]
			continue
		if hdulist[2].data.field('zwarning')[0] == 0 or hdulist[2].data.field('zwarning')[0] == 16:
			z = hdulist[2].data.field('z')[0]
			hdulist.close()
			try:
				hdulist = fits.open(path_SP+names[1]); hdulist.close()
			except IOError:
				missing += [(infile[jj]['sdss_name'],names[1])]
				continue
		else:
			z_issue+=[(infile[jj]['sdss_name'],names[0])]
			continue
		if z>0.41 and z<2.17:
			for name in names:
				hdulist = fits.open(path_SP+name)

		#		Find the coordinates of the object (degrees) from the fits file (used for checking object identity)
				ra, dec = hdulist[0].header['PLUG_RA'], hdulist[0].header['PLUG_DEC']
				coords+=[(ra,dec)]

		#		Include a measure of the S/N
				SN =  hdulist[2].data['sn_median_all'][0]

		#		Find average airmass of exposure
				airmass_list = []; exposure_times = []
				for ii in range(4,len(hdulist)):
					if hdulist[ii].header['EXTNAME'][0]=='B':
						airmass_list += [hdulist[ii].header['AIRMASS']]
						exposure_times += [hdulist[ii].header['EXPTIME']]

				if sum(exposure_times)==0:
					exposure += [(infile[jj]['sdss_name'],name)]
					if name==names[1]:
						del spec_list[-1],coords[-1]
					break
				else:
					a = np.average(airmass_list,weights=exposure_times)

				hdulist.close()
				spec_list += [ (name,infile[jj]['sdss_name'],infile[jj]['ra'],infile[jj]['dec'],z,a,SN,infile[jj]['A_v_sdss']) ]

			#Check the coordinates match, otherwise discard two spectra for the last object
			if coords[-1][0]-coords[-2][0]>0.05 or coords[-1][1]-coords[-2][1]>0.05:
				del coords[-2:]
				del spec_list[-2:]
				coord_mismatch += [names]

	print '##############################'
	print 'finished',mp.current_process(),len(spec_list)
	print 'Number of z-issues = {}; number of missing spectra = {}'.format(len(z_issue),len(missing),len(coord_mismatch))
	print 'Number of coordinate mismatches = {}; number of exposure issues= {}'.format(len(coord_mismatch),len(exposure))
	print 'missing', missing
	print 'exposure', exposure
	print '##############################'
	return spec_list

def add_extinction(ext,inp):
	'''Add extinctions, downloaded from the IRSA website'''
	if len(inp)!=len(ext)*2: print 'Error: sample list and extinction list do not match; sample {} & extinction = {}'.format(len(fi),len(ext)*2); return
	tmp=[]
	for ii in range(len(ext)):
		try:
			dt = abs(int(inp[2*ii]['spec_name'][10:15])-int(inp[2*ii+1]['spec_name'][10:15]))
			tmp += [ tuple(np.insert(list(inp[2*ii]),(5,-1),(dt,exp[ii]))),tuple(np.insert(list(inp[2*ii+1]),(5,-1),(dt,ext[ii]))) ]
		except:
			if inp[2*ii]['spec_name'][0]=='/':
				mjd1 = int(inp[2*ii]['spec_name'][12:17])
			else: mjd1 = int(inp[2*ii]['spec_name'][10:15])
			if inp[2*ii+1]['spec_name'][0]=='/':
				mjd2 = int(inp[2*ii+1]['spec_name'][12:17])
			else: mjd2 = int(inp[2*ii+1]['spec_name'][10:15])
			dt = abs(mjd1-mjd2)
			tmp += [ tuple(np.insert(list(inp[2*ii]),(5,-1),(dt,ext[ii]))),tuple(np.insert(list(inp[2*ii+1]),(5,-1),(dt,ext[ii]))) ]
	return tmp

def save_data(outfile,data,header=()):
	'''Write data to file'''
	with open(outfile,'w') as of:
		of.write('#')
		for ii in range(len(header)):
			of.write( '{}'.format(header[ii].ljust(30)) )
		of.write('\n')
		for row in data:
			of.write(' ')
			for el in row:
				of.write( '{}'.format(str(el).ljust(30)) )
			of.write('\n')
	print 'done'

def make_download_list(outfile,data):
	'''Write data to file in a format suitable for SDSS bulk download'''
	with open(outfile,'w') as of:
		for row in data:
			of.write(str(row['plate']).zfill(4)+'/spec-'+str(row['plate']).zfill(4)+'-'+str(row['mjd'])+'-'+str(row['fiber']).zfill(4)+'.fits\n')
		print 'done'

####################################

t_b = datetime.datetime.now()


##step 1: create download list and matching list of objects names (Nspec>= 2 printed and delta_t>30 days for final list)
#inp_split = split_input(T14)
#p=mp.Pool(processes=ncpu)
#dr14_mult = p.map(if_mult_obs,inp_split)
#dr14_mult = [ii for sub in dr14_mult for ii in sub]
#dl_list = []; spec_list = []; dr14_mult_list=[]
#for row in dr14_mult:
#	dl_list += [(row[1],row[2],row[3]),(row[4],row[5],row[6])]
#	spec_list += [(row[0],row[7],row[8],round(row[9],3),row[10],round(row[11],3),'spec-'+str(row[1]).zfill(4)+'-'+str(row[2])+'-'+str(row[3]).zfill(4)+'.fits'),(row[0],row[7],row[8],round(row[9],3),row[10],round(row[11],3),'spec-'+str(row[4]).zfill(4)+'-'+str(row[5])+'-'+str(row[6]).zfill(4)+'.fits')]
#dl_list = np.array(dl_list,dtype=[('plate',int),('mjd',int),('fiber',int)])
#spec_list = np.array(spec_list,dtype=[('sdss_name','S20'),('ra',float),('dec',float),('z',float),('bal_flag',int),('A_v_sdss',float),('spec_name','S30')])
#save_data('DR14_spec_list',spec_list,header=spec_list.dtype.names)
#make_download_list('DR14_DL_list',dl_list)

##step 2: remove BALQs
#spec_list = np.loadtxt('DR14_spec_list',dtype=[('sdss_name','S20'),('ra',float),('dec',float),('z',float),('bal_flag',int),('A_v_sdss',float),('spec_name','S30')])
#dr14_mult_balqrm = remove_balq(T7S,spec_list)
#save_data('DR14_spec_list_bal',dr14_mult_balqrm,header=dr14_mult_balqrm.dtype.names)

##step 3: filter for z-warning and MgII range; create mgii_sample file and coord_list
#dr14_mult_balqrm = np.loadtxt('DR14_spec_list_bal',usecols=(0,1,2,3,5,6),dtype=[('sdss_name','S20'),('ra',float),('dec',float),('z',float),('A_v_sdss',float),('spec_name','S30')])
#part_fz = partial(filter_z,path_SP)
#inp_split = split_input(dr14_mult_balqrm)
#p = mp.Pool(processes=ncpu)
#sample_list = p.map(part_fz,inp_split)
#sample_list = [ii for sub in sample_list for ii in sub]
#sample_list = np.array(sample_list,dtype=[('spec_name','S30'),('sdss_name','S20'),('ra',float),('dec',float),('z_est',float),('airmass',float),('S/N',float),('A_v_sdss',float)])
#save_data('mgii_sample_list',sample_list,header=sample_list.dtype.names) #intermediate save file
#sample_list = np.loadtxt('mgii_sample_list',dtype=[('spec_name','S30'),('sdss_name','S20'),('ra',float),('dec',float),('z_est',float),('airmass',float),('S/N',float),('A_v_sdss',float)])
#coord_list = []
#row_num = 0.
#for row in sample_list:
#	row_num+=1
#	if row_num%2!=0: continue
#	coord_list+=[(row['ra'],row['dec'],'1')]
#np.savetxt('mgii_coord_list',np.array(coord_list),fmt='%15s   %15s   %4s',header='|  ra  |  dec  |  size  |\n| double | double | double |')

#step 4: add A_v for Schlafy & Finkbeiner model, as well as delta_t (the A_v values need to be downloaded from the IRSA website)
#sample_list = np.loadtxt('mgii_sample_list',dtype=[('spec_name','S30'),('sdss_name','S20'),('ra',float),('dec',float),('z_est',float),('airmass',float),('S/N',float),('A_v_sdss',float)])
#ext = np.loadtxt(path+'SpectraDL/mgii_extinction',usecols=8) #Using the 8th columns assumes IRSA formatting
#sample_list = add_extinction(ext,sample_list)
#sample_list = np.array(sample_list,dtype=[('spec_name','S30'),('sdss_name','S20'),('ra',float),('dec',float),('z_est',float),('delta_t',int),('airmass',float),('S/N',float),('A_v_SF',float),('A_v_sdss',float)])
#save_data = save_data('mgii_sample_ext_list',sample_list,header=sample_list.dtype.names)

print 'Finished in: ',str(datetime.datetime.now()-t_b)
sys.exit()

####################################





