import numpy as np
from nbodykit.lab import *
import matplotlib.pyplot as plt
import os
from natsort import realsorted, ns
import h5py as h5
from nbodykit.algorithms.paircount_tpcf.tpcf import SimulationBox2PCF
from nbodykit.source.catalog import ArrayCatalog
import matplotlib
from time import time
from argparse import ArgumentParser
from nbodykit import CurrentMPIComm
import illustris_python as il
from matplotlib import colors
import shutil
from myFOF import FOF as myfof
import matplotlib.gridspec as gridspec
import seaborn as sns

#make parser
parser = ArgumentParser(description='create statistics')

parser.add_argument('--counts', action='store_true', help='eval stat for counts fields')
parser.add_argument('--density', action='store_true', help='eval stat for density fields')
parser.add_argument('--overdensity', action='store_true', help='eval stat for overdensity fields')
parser.add_argument('--run-name', type=str, default='no_name', help='enter name of run to test')
parser.add_argument('--cube-sz', type=int, default=64, help='number of subcube voxels per dimension')
parser.add_argument('--pixside', type=int, default=2048, help='number of sim voxels per dimension')
parser.add_argument('--state-num', type=int, default=1, help='number of state to use, eg. 600 for state_600.pt')

args = parser.parse_args()

run = args.run_name
statenum = args.state_num

print(run, flush=True)

direc = '/scratch/ds6311/mywork/statistics/'+run+'/state_'+str(statenum)+'/'

#create dirs
if not os.path.exists(direc):
    os.makedirs(direc)
    
direc_data = '/scratch/ds6311/mywork/statistics/'+run+'/state_'+str(statenum)+'/data/'

if not os.path.exists(direc_data):
    os.makedirs(direc_data)

direc_plots = '/scratch/ds6311/mywork/statistics/'+run+'/state_'+str(statenum)+'/plots/'

if not os.path.exists(direc_plots):
    os.makedirs(direc_plots)

run_path = '/scratch/ds6311/mywork/scripts/test/'+run+'/state_'+str(statenum)+'/' #batch folder

########################################################

comm = CurrentMPIComm.get()
cosmo = cosmology.WMAP9

#various functions (some unused!)

def myexp(x):
    return np.exp(x) - 1e-7
#log not log1p
def mylog(x):
    return np.log(1e-7 + x)
def mylog1p(x):
    return np.log1p(1e-7 + x)
def poissonSample(cub):
    floor = np.floor(cub)
    diff = cub - floor
    rand = np.random.rand(cub.shape[0],cub.shape[1],cub.shape[2],cub.shape[3],cub.shape[4])
    return np.where(rand < diff, floor + 1, floor).astype('int')
def unmakeDensity(cub):
    dm_particle_mass1 = 7.5e6 #sunmasses, 7.5e6 for Ill-1-Dark, 6.0e7 for Ill-2-Dark
    dm_particle_mass2 = 6e7 #sunmasses, 7.5e6 for Ill-1-Dark, 6.0e7 for Ill-2-Dark
    l_box = 75 #Mpc/h
    why = np.copy(cub)
    if why.shape[0] == 3:
        why[0] /= dm_particle_mass2/(l_box/pixside)**3 #convert to density Ill2
        why[1] /= dm_particle_mass1/(l_box/pixside)**3 #convert to density Ill1
        why[2] /= dm_particle_mass1/(l_box/pixside)**3 #convert to density Ill1
    else:
        print('Weird input (not shape input/output/target)')
        why /= dm_particle_mass1/(l_box/pixside)**3 #convert to density
    return why
def unmakeOverdensity(cub): #input overdens, output number count
    mean_dens = 107174542222. #from Illustris 1 AND 2 (they have the same)
    dens = np.copy(cub)
    dens = dens * mean_dens + mean_dens
    return unmakeDensity(dens)
def getDensity(cub, field): #number count input
    l_box = 75 #Mpc/h
    if field == 0:
        return cub * 6e7/(l_box/pixside)**3  #Illustris 2
    else:
        return cub * 7.5e6/(l_box/pixside)**3  #Illustris 1
def getOverdensity(cub, field): #count input
    dens = getDensity(cub, field) 
    mean_dens = np.mean(dens)
    return (dens - mean_dens) / mean_dens
def getOverdensityfromDensity(cub, field): #count input
    dens = cub
    mean_dens = np.mean(dens)
    return (dens - mean_dens) / mean_dens
def getPS(cat, pixelside, box_sz):
    """
    cube -> mesh -> power spectrum
    """
    r = FFTPower(cat, mode='1d', Nmesh=pixelside, BoxSize=box_sz)
    return r.power
def getCorr(cub, pixelside):
    """
    cube -> mesh -> power spectrum -> corr fct
    """
    mesh = ArrayMesh(cub, BoxSize=75/pixside*pixelside)
    c = FFTCorr(mesh, mode='1d')
    return c.corr
def getPDF(cub): #number count input
    hist, edges = np.histogram(cub, bins=np.arange(int(cub.max())+2), density=True) #'+2' because we want cub.max() entry also to be part of hist
    histn, edgesn = np.histogram(cub, bins=np.arange(int(cub.max())+2), density=False)
    edges = edges[:-1] #remove last edge
    edgesn = edgesn[:-1] #remove last edge
    totN = np.sum(histn*edgesn)
    totZero = histn[0]
    return hist, edges, float(totN), float(totZero)

#PLOT COLORS

inn_c = '#FFA3AF'
out_c = '#007CBE'
tgt_c = '#00AF54'

col = [inn_c, out_c, tgt_c]

ill3_c = '#FFD639'
ill2_c = '#FBA300'

col_ill = [ill3_c, ill2_c]


#################################################################
#------------- import and combine cpus cat data to cat ----------
#################################################################

box_sz = 75./2 #for testing cube
pixside = args.pixside
testcube_sz = int(pixside/2)
cubesz = args.cube_sz

data_dir = direc_data+'cpu_cats/'

f = []
for (dirpath, dirnames, filenames) in os.walk(data_dir): #scan all data numpy cats
    f.extend(filenames)

data = [np.empty(0, dtype=[('Position', ('f8', 3))]), np.empty(0, dtype=[('Position', ('f8', 3))]), np.empty(0, dtype=[('Position', ('f8', 3))])]    
    
for name in f:
    data_cat = np.load(data_dir+name, allow_pickle=True)
    for i in range(3):
        data[i] = np.concatenate((data[i],data_cat[i]))
del f

#################################################################
#-------------------- CALC power & catalogs ---------------------
#---------- (2pcf calc is sepreate on multiple ranks) -----------
#################################################################


print('args.overdensity:', args.overdensity)
print('args.density:', args.density)
print('args.counts:', args.counts)

#save full catalogs
cat0 = ArrayCatalog(data[0])
cat1 = ArrayCatalog(data[1])
cat2 = ArrayCatalog(data[2])

cats = [cat0, cat1, cat2]

cat0.save(direc_data+'Cat_Input')
cat1.save(direc_data+'Cat_Output')
cat2.save(direc_data+'Cat_Target')
#get random subsamples of catalogs
sub_perc = 0.2 #percent
sub = [int(cat0.size * sub_perc), int(cat1.size * sub_perc), int(cat2.size * sub_perc)]

sub_data = [numpy.empty(sub[0], dtype=[('Position', ('f8', 3))]), numpy.empty(sub[1], dtype=[('Position', ('f8', 3))]), numpy.empty(sub[2], dtype=[('Position', ('f8', 3))])]
sub_idx = [0,0,0]
for i in range(3):
    sub_idx[i] = np.random.randint(0, data[i]['Position'].shape[0], size=sub[i])
for i in range(3):
    sub_data[i]['Position'] = data[i]['Position'][sub_idx[i]]
    
sub0 = ArrayCatalog(sub_data[0])
sub1 = ArrayCatalog(sub_data[1])
sub2 = ArrayCatalog(sub_data[2])

sub0.save(direc_data+'Cat_Input_sub20')
sub1.save(direc_data+'Cat_Output_sub20')
sub2.save(direc_data+'Cat_Target_sub20')

#PS
basePath3='/scratch/ds6311/Illustris-3/dm_only'
basePath2='/scratch/ds6311/Illustris-2/dm_only'
#load illustris dm pos:
particles3 = il.snapshot.loadSubset(basePath3,135,'dm',['Coordinates']) / 1e3 #in Mpc/h
particles2 = il.snapshot.loadSubset(basePath2,135,'dm',['Coordinates']) / 1e3 #in Mpc/h
#select testing cube portion
particles3 = particles3[(particles3[:,0] >= box_sz) & (particles3[:,1] >= box_sz) & (particles3[:,2] >= box_sz)]
particles2 = particles2[(particles2[:,0] >= box_sz) & (particles2[:,1] >= box_sz) & (particles2[:,2] >= box_sz)]

partss = [np.empty(particles3.shape[0], dtype=[('Position', ('f8', 3))]), np.empty(particles2.shape[0], dtype=[('Position', ('f8', 3))])]

partss[0]['Position'] = particles3
partss[1]['Position'] = particles2

parts3 = ArrayCatalog(partss[0])
parts2 = ArrayCatalog(partss[1])

parts = [parts3, parts2]

del particles3, particles2, partss

label = ['low-res','generated','high-res']
label_il = ['Illustris-3', 'Illustris-2']

ls = ['dashed', 'solid', 'dashdot']

import seaborn as sns
sns.set_theme(context='poster', style='white', palette='Set2')

alpha = 0.7
lw = 6

fig = plt.figure(figsize=(12,12))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

params = {'mathtext.default': 'regular',
        'font.family': 'serif',
        }          
plt.rcParams.update(params)

ax0 = plt.subplot(gs[0,0])
for i in range(3): #training data power
    ax0.loglog(Pks[i]['k'][1:], Pks[i]['power'][1:].real, label=label[i], linestyle='solid', alpha=alpha, linewidth=lw, color=col[i])
for i in range(2): #illustris catalog power
    ax0.loglog(Pks[i+3]['k'][1:], Pks[i+3]['power'][1:].real, label=label_il[i], linestyle='dashed', alpha=alpha, linewidth=lw, color=col_ill[i])
#plt.vlines(2*np.pi/(box_sz/testcube_sz), 0, 1, transform=plt.get_xaxis_transform(), color='gray', alpha=0.5)
ax0.set_ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]", labelpad=11)
#ax0.set_title("FFT Power Spectrum")
ax0.legend()
# plot Nyquist frequency
k_ny = np.pi * testcube_sz / box_sz
plt.axvline(x=k_ny, c='k', ls='dashed', alpha=0.7)
ax0.legend()
ax0.grid(True, alpha=0.3)
ax0.tick_params(axis='x', labeltop=False, bottom=False)
ax0.tick_params(axis='y', which='both', left=True)

ax1 = plt.subplot(gs[1,0], sharex = ax0)

plt.setp(ax0.get_xticklabels(), visible=False)
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
ax1.tick_params(labeltop=False, top=True)
ax1.set_ylabel(r"$P \ / \ P_{\mathrm{Illustris2}}$", labelpad=35)
ax1.set_xlabel(r"$\mathrm{k}$ [$\mathrm{h} \ \mathrm{Mpc}^{-1}$]")
ax1.semilogx(Pks[0]['k'][1:], Pks[0]['power'][1:].real / Pks[4]['power'][1:].real, label=label[0], ls='solid', color=inn_c, alpha=alpha, linewidth=lw)
ax1.semilogx(Pks[1]['k'][1:], Pks[1]['power'][1:].real / Pks[4]['power'][1:].real, label=label[1], ls='solid', color=out_c, alpha=alpha, linewidth=lw)
ax1.semilogx(Pks[2]['k'][1:], Pks[2]['power'][1:].real / Pks[4]['power'][1:].real, label=label[2], ls='solid', color=tgt_c, alpha=alpha, linewidth=lw)
ax1.semilogx(Pks[3]['k'][1:], Pks[3]['power'][1:].real / Pks[4]['power'][1:].real, label=label_il[0], ls='dashed', color=ill3_c, alpha=alpha, linewidth=lw)
ax1.semilogx(Pks[4]['k'][1:], Pks[4]['power'][1:].real / Pks[4]['power'][1:].real, label=label_il[0], ls='dashed', color=ill2_c, alpha=alpha, linewidth=lw)
# plot Nyquist frequency
k_ny = np.pi * testcube_sz / box_sz
plt.axvline(x=k_ny, c='k', ls='dashed', alpha=0.7)
plt.setp(ax0.get_xticklabels(), visible=False) #remove tick labels
ax1.tick_params(axis='x',labeltop=False, bottom=True, top=True, which='both')
ax1.tick_params(axis='y', which='both', left=True)
ax1.grid(True, alpha=0.3)


#add Nyquist tick and label
lim = ax1.get_xlim()
extraticks = [k_ny]
ax1.set_xticks(list(ax1.get_xticks()) + extraticks)
ax1.set_xlim(lim)

plt.draw()

labels = [w.get_text() for w in ax1.get_xticklabels()]
labels[-1] = r'$k_{Nyq}$'
ax1.set_xticklabels(labels)
#offset tick label
dx = -10/72.; dy = -10/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
lab = ax1.xaxis.get_majorticklabels()[-1]
lab.set_transform(lab.get_transform() + offset)

plt.savefig(direc_plots+'fftPower.pdf')

del Pks

#create .hdf5 cats for eg. Rockstar halo finder
catnames = ['Input', 'Output', 'Target']

for name in catnames:
    cat_path = direc_data+'Cat_'+name+'/'
    pos = BigFileCatalog(cat_path, header='Header')['Position']
    pos = np.array(pos)
    
    num_part = pos.shape[0]
    if name == 'Input':
        partmass = 338857141.965 #ill-3
    else:
        partmass = 42357142.7457 #ill-2
    
    with h5.File(direc_data+'cat_'+name+'.hdf5', "w") as f:
        header = f.create_group("Header")
        header.attrs['BoxSize'] = box_sz
        header.attrs['MassTable'] = [0, partmass, 0, 0, 0, 0]
        header.attrs['NumPart_ThisFile'] = [0, num_part, 0, 0, 0, 0]
        header.attrs['NumPart_Total'] = [0, int(num_part % (2**32)), 0, 0, 0, 0]
        header.attrs['NumPart_Total_HighWord'] = [0, int(np.floor(num_part / (2**32))), 0, 0, 0, 0]
        header.attrs['Omega0'] = 0.2726
        header.attrs['OmegaLambda'] = 0.7274
        header.attrs['Redshift'] = 2.220446049250313e-16
        header.attrs['Time'] = 0.9999999999999998
        header.attrs['HubbleParam'] = 0.704 #h0
        header.attrs['NumFilesPerSnapshot'] = 1
        header.attrs['Composition_vector_length'] = 0
        header.attrs['Flag_Cooling'] = 0
        header.attrs['Flag_DoublePrecision'] = 0
        header.attrs['Flag_Feedback'] = 0
        header.attrs['Flag_Metals'] = 0
        header.attrs['Flag_Sfr'] = 0
        header.attrs['Flag_StellarAge'] = 0

        parttype1 = f.create_group("PartType1")
        parttype1.create_dataset("Coordinates", (num_part,3), dtype='f4', data=pos)
        parttype1.create_dataset("ParticleIDs", (num_part,), dtype='u8', data=np.arange(num_part))
        parttype1.create_dataset("Potential", (num_part,), dtype='f4', data=np.zeros(num_part))
        parttype1.create_dataset("Velocities", (num_part,3), dtype='f4', data=np.zeros((num_part,3)))

#PDF (uses a lot of memory with this simple implementation!)
#get list of cubenames + dir again
lis = []
h = []
for (dirpath, dirnames, filenames) in os.walk(run_path):
    h.extend(dirnames)
for direc in h:
    f = []
    path = run_path+direc+'/'
    for (dirpath, dirnames, filenames) in os.walk(path):
        for i in range(len(filenames)):
            lis.append(dirpath+'/'+filenames[i])

sub_lis = np.random.choice(lis, size=int(len(lis)*1)) #take 100% of all testing cubes
cubes_01 = np.zeros((3, sub_lis.size, cubesz, cubesz, cubesz))
for j in range(3):
    for i in range(sub_lis.size):
        cubes_01[j][i] = np.load(sub_lis[i])[label[j]][0,0]
        
cubes_01 = cubes_01.clip(min=0)
cubes_01 = poissonSample(cubes_01)

bins = [20, 40, 40]

sns.set_theme(context='poster', style='white', palette='Set2')

params = {'mathtext.default': 'regular',
        'font.family': 'serif',
        }          
plt.rcParams.update(params)

bins = [20, 40, 40]
label_plot = ['low-res', 'generated', 'high-res']
fig = plt.figure(figsize=(12,12))
plt.grid(True, alpha=0.3)
#plt.title('PDF comparison')
alpha=1.
for i in range(3):
    if i > 0: alpha = 0.3
    plt.hist(cubes_01[i].flatten(), label=label_plot[i], bins=bins[i], alpha=alpha, histtype='stepfilled', color=col[i])
plt.yscale('log')
plt.ylabel('# of voxels')
plt.xlabel('count value')
plt.tick_params(axis='x', labeltop=False, bottom=True)
plt.tick_params(axis='y', which='both', left=True)
plt.legend()

plt.savefig(direc_plots+'pdf_comparison.pdf')
    
    
del cubes_01, sub_lis, fig


#################################################################
#------------------------- CALC FOF -----------------------------
#################################################################

cube_sz = testcube_sz

inn = BigFileCatalog(direc_data+'Cat_Input')
out = BigFileCatalog(direc_data+'Cat_Output')
tgt = BigFileCatalog(direc_data+'Cat_Target')

low_hmass_limit = 1e8

h = 0.704 #from wmap9
part_mass2 = 0.0338857141965349 * 1e10 / h #for Ill-3, in sun masses
part_mass = 0.00423571427456686 * 1e10 / h #for Ill-2, in sun masses

print('TOTAL NUMBER OF PARTS IN INPUT CATALOG: {:.4e}'.format(inn['Position'].size))
print('TOTAL NUMBER OF PARTS IN OUTPUT CATALOG: {:.4e}'.format(out['Position'].size))
print('TOTAL NUMBER OF PARTS IN TARGET CATALOG: {:.4e}'.format(tgt['Position'].size))



inn.attrs['BoxSize'] = [box_sz, box_sz, box_sz] #Mpc/h
inn.attrs['Nmesh'] = [cube_sz*2, cube_sz*2, cube_sz*2] #number of mesh cells per side of box ??? what exactly is that?
out.attrs['BoxSize'] = [box_sz, box_sz, box_sz] #Mpc/h
out.attrs['Nmesh'] = [cube_sz*2, cube_sz*2, cube_sz*2] #number of mesh cells per side of box ??? what exactly is that?
tgt.attrs['BoxSize'] = [box_sz, box_sz, box_sz] #Mpc/h
tgt.attrs['Nmesh'] = [cube_sz*2, cube_sz*2, cube_sz*2] #number of mesh cells per side of box ??? what exactly is that?
inn['Velocity'] = list(np.zeros(inn['Position'].shape, dtype='int'))
out['Velocity'] = list(np.zeros(out['Position'].shape, dtype='int'))
tgt['Velocity'] = list(np.zeros(tgt['Position'].shape, dtype='int'))

linking_length = 0.2 #in relative units
min_halo = 32 #min number of particles for halo

friends_inn = myfof(inn, linking_length, min_halo, periodic=False)
friends_out = myfof(out, linking_length, min_halo, periodic=False)
friends_tgt = myfof(tgt, linking_length, min_halo, periodic=False)

fof_inn = friends_inn.to_halos(part_mass2, cosmo, 0)
fof_out = friends_out.to_halos(part_mass, cosmo, 0)
fof_tgt = friends_tgt.to_halos(part_mass, cosmo, 0)

#save cats
fof_inn.save(direc_data+'FoFCat_Input')
inn[friends_inn.labels != 0].save(direc_data+'Cat_Input_onlyFoFparts')
fof_out.save(direc_data+'FoFCat_Output')
out[friends_out.labels != 0].save(direc_data+'Cat_Output_onlyFoFparts')
fof_tgt.save(direc_data+'FoFCat_Target')
tgt[friends_tgt.labels != 0].save(direc_data+'Cat_Target_onlyFoFparts')

#mask and sort by masses
cat_mass_inn = fof_inn['Mass']
cat_mass_out = fof_out['Mass']
cat_mass_tgt = fof_tgt['Mass']

mask_inn = cat_mass_inn >= low_hmass_limit
masked_inn = np.array(cat_mass_inn[mask_inn])

mask_out = cat_mass_out >= low_hmass_limit
masked_out = np.array(cat_mass_out[mask_out])

mask_tgt = cat_mass_tgt >= low_hmass_limit
masked_tgt = np.array(cat_mass_tgt[mask_tgt])

#-----------------------------------

bins = np.logspace(9., 12., 25)
V_ill = 75.**3 #Mpc/h^3
V_box = (75./2)**3

def getHalomass(bins, m_mass):
    n = np.zeros(bins.shape[0])
    for i, binn in enumerate(bins):
        for k in range(m_mass.shape[0]):
            if m_mass[k] < binn:
                n[i] = k
                break
    return bins, n

#import Ill-3 groupcat and get halos
basePath = '/scratch/ds6311/Illustris-3/groupcat'

halos = il.groupcat.loadHalos(basePath,135)

#create mass entry
halos['Mass'] = halos['GroupMass'] * 1e10 / h #halos['GroupMass'] in 10^10 𝑀⊙/ℎ

mass_mask = halos['Mass'] >= low_hmass_limit
masked_mass2 = halos['Mass'][mass_mask]
masked_mass2[::-1].sort()

bins, number_ill2 = getHalomass(bins,masked_mass2)
number_ill2_frac = number_ill2 / V_ill #divide by volume


#import Ill-2 groupcat and get halos
basePath = '/scratch/ds6311/Illustris-2/groupcat'

halos = il.groupcat.loadHalos(basePath,135)

#create mass entry
halos['Mass'] = halos['GroupMass'] * 1e10 / h #halos['GroupMass'] in 10^10 𝑀⊙/ℎ

mass_mask = halos['Mass'] >= low_hmass_limit
masked_mass1 = halos['Mass'][mass_mask]
masked_mass1[::-1].sort()

bins, number_ill1 = getHalomass(bins,masked_mass1)
number_ill1_frac = number_ill1 / V_ill #divide by volume

#------------------------------------

bins, number_fof_inn = getHalomass(bins, masked_inn)
number_fof_inn_frac = number_fof_inn / V_box

bins, number_fof_out = getHalomass(bins, masked_out)
number_fof_out_frac = number_fof_out / V_box

bins, number_fof_tgt = getHalomass(bins, masked_tgt)
number_fof_tgt_frac = number_fof_tgt / V_box

#------------------------------------

#get differential halo mass functions

def dNdM(bins, m_mass):
    hist = np.histogram(m_mass, bins=bins)[0] #dN entries
    diff = np.zeros(hist.shape[0])
    for i in range(hist.shape[0]): #dM entries
        diff[i] = bins[i+1] - bins[i]
    return hist/diff #right-most edge does not have hist entry due to definition
def dNdlogM(bins, m_mass):
    hist = np.histogram(m_mass, bins=bins)[0] #dN entries
    diff = np.zeros(hist.shape[0])
    for i in range(hist.shape[0]): #dM entries
        diff[i] = bins[i+1] - bins[i]
    return hist/np.log(diff) #right-most edge does not have hist entry due to definition

bins_diff = np.logspace(8., 15., 100)


diff_ill2 = dNdM(bins_diff, masked_mass2) #Ill-2 groupcat
diff_ill2_frac = diff_ill2 / V_ill

diff_ill1 = dNdM(bins_diff, masked_mass1) #Ill-1 groupcat
diff_ill1_frac = diff_ill1 / V_ill

diff_inn = dNdM(bins_diff, masked_inn)
diff_inn_frac = diff_inn / V_box

diff_out = dNdM(bins_diff, masked_out)
diff_out_frac = diff_out / V_box

diff_tgt = dNdM(bins_diff, masked_tgt)
diff_tgt_frac = diff_tgt / V_box


####

diff_ill2_log = dNdlogM(bins_diff, masked_mass2) #Ill-2 groupcat
diff_ill2_frac_log = diff_ill2_log / V_ill

diff_ill1_log = dNdlogM(bins_diff, masked_mass1) #Ill-1 groupcat
diff_ill1_frac_log = diff_ill1_log / V_ill

diff_inn_log = dNdlogM(bins_diff, masked_inn)
diff_inn_frac_log = diff_inn_log / V_box

diff_out_log = dNdlogM(bins_diff, masked_out)
diff_out_frac_log = diff_out_log / V_box

diff_tgt_log = dNdlogM(bins_diff, masked_tgt)
diff_tgt_frac_log = diff_tgt_log / V_box


###################################################
# PLOT

alpha = 0.7
lw = 6

sns.set_theme(context='poster', style='white', palette='Set2')

fig = plt.figure(figsize=(12,12))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

params = {'mathtext.default': 'regular',
        'font.family': 'serif',
        }          
plt.rcParams.update(params)

ax0 = plt.subplot(gs[0])
ax0.grid(True, alpha=0.3)
#ax0.set_title("diff hmf")
ax0.loglog(bins_diff[:-1], diff_inn_frac_log, alpha=alpha, label='low-res', color=inn_c, linewidth=lw)
ax0.loglog(bins_diff[:-1], diff_out_frac_log, alpha=alpha, label='generated', color=out_c, linewidth=lw)
ax0.loglog(bins_diff[:-1], diff_tgt_frac_log, alpha=alpha, label='high-res', color=tgt_c, linewidth=lw)
ax0.loglog(bins_diff[:-1], diff_ill2_frac_log, alpha=alpha, label='Illustris-3', ls='dashed', color=ill3_c, linewidth=lw)
ax0.loglog(bins_diff[:-1], diff_ill1_frac_log, alpha=alpha, label='Illustris-2', ls='dashed', color=ill2_c, linewidth=lw)
ax0.set_ylabel(r'$dN \ / \ dlogM$', labelpad=5)
ax0.legend()
ax0.tick_params(axis='x', labeltop=False, bottom=False)
ax0.tick_params(axis='y', which='both', left=True)

ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.set_ylabel('fraction', labelpad=14)
ax1.grid(True, alpha=0.3)
ax1.set_xlabel(r'$M_{\rm halo}/M_{\odot}$')
ax1.semilogx(bins_diff[:-1], diff_inn_frac_log / diff_ill1_frac_log, alpha=alpha, color=inn_c, linewidth=lw)
ax1.semilogx(bins_diff[:-1], diff_out_frac_log / diff_ill1_frac_log, alpha=alpha, color=out_c, linewidth=lw)
ax1.semilogx(bins_diff[:-1], diff_tgt_frac_log / diff_ill1_frac_log, alpha=alpha, color=tgt_c, linewidth=lw)
ax1.semilogx(bins_diff[:-1], diff_ill2_frac_log / diff_ill1_frac_log, alpha=alpha, ls='dashed', color=ill3_c, linewidth=lw)
ax1.semilogx(bins_diff[:-1], diff_ill1_frac_log / diff_ill1_frac_log, alpha=alpha, ls='dashed', color=ill2_c, linewidth=lw)
plt.setp(ax0.get_xticklabels(), visible=False) #remove tick labels
ax1.tick_params(axis='x',labeltop=False, bottom=True, top=True, which='both')
ax1.tick_params(axis='y', which='both', left=True)

fig.subplots_adjust(hspace=0)

plt.savefig(direc_plots+'dNdlogM_HMF_comparison.pdf')


##########################################
#---------- plot halo 2pcf ---------------
##########################################

bins = np.array([1e10, 4e10, 1.2e11])

sns.set_theme(context='poster', style='white', palette='Set2')

params = {'mathtext.default': 'regular',
        'font.family': 'serif',
        }          
plt.rcParams.update(params)


lowlimit = bins[0]
highlimit = bins[1]

mask_inn = (cat_mass_inn >= lowlimit) & (cat_mass_inn < highlimit)
masked_pos_inn = fof_inn['Position'][mask_inn]

mask_out = (cat_mass_out >= lowlimit) & (cat_mass_out < highlimit)
masked_pos_out = fof_out['Position'][mask_out]

mask_tgt = (cat_mass_tgt >= lowlimit) & (cat_mass_tgt < highlimit)
masked_pos_tgt = fof_tgt['Position'][mask_tgt]

data0 = {}
data0['Position'] = np.array(masked_pos_inn)
n_inn = data0['Position'].shape[0]
rand0 = {}
rand0['Position'] = np.random.uniform(0, box_sz, size=(n_inn*10,3))
data1 = {}
data1['Position'] = np.array(masked_pos_out)
n_out = data1['Position'].shape[0]
rand1 = {}
rand1['Position'] = np.random.uniform(0, box_sz, size=(n_out*10,3))
data2 = {}
data2['Position'] = np.array(masked_pos_tgt)
n_tgt = data2['Position'].shape[0]
rand2 = {}
rand2['Position'] = np.random.uniform(0, box_sz, size=(n_tgt*10,3))

print(data2['Position'].min(),data2['Position'].max())
print(rand2['Position'].min(),rand2['Position'].max())

cat0 = ArrayCatalog(data0, comm=comm)
cat_rand0 = ArrayCatalog(rand0, comm=comm)
cat1 = ArrayCatalog(data1, comm=comm)
cat_rand1 = ArrayCatalog(rand1, comm=comm)
cat2 = ArrayCatalog(data2, comm=comm)
cat_rand2 = ArrayCatalog(rand2, comm=comm)

#calc auto corr
auto_bins = np.logspace(-1, 1, 25) #in Mpc/h

cf0 = SimulationBox2PCF('1d', data1=cat0, randoms1=cat_rand0, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)
cf1 = SimulationBox2PCF('1d', data1=cat1, randoms1=cat_rand1, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)
cf2 = SimulationBox2PCF('1d', data1=cat2, randoms1=cat_rand2, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)

#illustris data catalog
#import Ill-3 groupcat and get halos
basePath2 = '/scratch/ds6311/Illustris-3/groupcat/'

h = 0.704 #from wmap9

halos2 = il.groupcat.loadHalos(basePath2,135)
halos2['Mass'] = halos2['GroupMass'] * 1e10 / h #halos['GroupMass'] in 10^10 𝑀⊙/ℎ
mass_mask2 = (halos2['Mass'] >= lowlimit) & (halos2['Mass'] < highlimit) & (halos2['GroupPos'][:,0] >= box_sz*1e3) & (halos2['GroupPos'][:,1] >= box_sz*1e3) & (halos2['GroupPos'][:,2] >= box_sz*1e3)

masked_pos2 = halos2['GroupPos'][mass_mask2,:] / 1e3 - box_sz #convert to Mpc/h

#import Ill-2 groupcat and get halos
basePath1 = '/scratch/ds6311/Illustris-2/groupcat/'

halos1 = il.groupcat.loadHalos(basePath1,135)
halos1['Mass'] = halos1['GroupMass'] * 1e10 / h #halos['GroupMass'] in 10^10 𝑀⊙/ℎ
mass_mask1 = (halos1['Mass'] >= lowlimit) & (halos1['Mass'] < highlimit) & (halos1['GroupPos'][:,0] >= box_sz*1e3) & (halos1['GroupPos'][:,1] >= box_sz*1e3) & (halos1['GroupPos'][:,2] >= box_sz*1e3)

masked_pos1 = halos1['GroupPos'][mass_mask1,:] / 1e3 - box_sz

data_ill2 = {}
data_ill2['Position'] = np.array(masked_pos2)
n_ill2 = data_ill2['Position'].shape[0]
rand_ill2 = {}
rand_ill2['Position'] = np.random.uniform(0, box_sz, size=(n_ill2*10,3))
data_ill1 = {}
data_ill1['Position'] = np.array(masked_pos1)
n_ill1 = data_ill1['Position'].shape[0]
rand_ill1 = {}
rand_ill1['Position'] = np.random.uniform(0, box_sz, size=(n_ill1*10,3))

print(data_ill1['Position'].min(),data_ill1['Position'].max())
print(rand_ill1['Position'].min(),rand_ill1['Position'].max())

cat_ill2 = ArrayCatalog(data_ill2, comm=comm)
cat_rand_ill2 = ArrayCatalog(rand_ill2, comm=comm)
cat_ill1 = ArrayCatalog(data_ill1, comm=comm)
cat_rand_ill1 = ArrayCatalog(rand_ill1, comm=comm)

cf_ill2 = SimulationBox2PCF('1d', data1=cat_ill2, randoms1=cat_rand_ill2, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)
cf_ill1 = SimulationBox2PCF('1d', data1=cat_ill1, randoms1=cat_rand_ill1, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)

alpha = 0.7
lw = 6

#plot
llim = 4
fig = plt.figure(figsize=(12,12))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
ax1 = plt.subplot(gs[0])
ax1.grid(True, alpha=0.3)
#ax1.set_title('Plot ('+str(i)+')')
ax1.loglog(cf0.corr['r'][llim:], cf0.corr['corr'][llim:], label='input', alpha=alpha, color=inn_c, linewidth=lw)
ax1.loglog(cf1.corr['r'][llim:], cf1.corr['corr'][llim:], label='output', alpha=alpha, color=out_c, linewidth=lw)
ax1.loglog(cf2.corr['r'][llim:], cf2.corr['corr'][llim:], label='target', alpha=alpha, color=tgt_c, linewidth=lw)
ax1.loglog(cf_ill2.corr['r'][llim:], cf_ill2.corr['corr'][llim:], label='Illustris-3', ls='dashdot', alpha=alpha, color=ill3_c, linewidth=lw)
ax1.loglog(cf_ill1.corr['r'][llim:], cf_ill1.corr['corr'][llim:], label='Illustris-2', ls='dashed', alpha=alpha, color=ill2_c, linewidth=lw)
ax1.legend()
ax1.set_ylabel(r'$\zeta_2 (r) \ \{$%.0e $\leq M_{\rm halo} <$%.0e $M_{\odot} \}$' % (lowlimit, highlimit), labelpad=10)
fig.add_subplot(ax1)
ax1.tick_params(axis='x', labeltop=False, bottom=False)
ax1.tick_params(axis='y', which='both', left=True)

ax2 = plt.subplot(gs[1], sharex=ax1)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.grid(True, alpha=0.3)
ax2.semilogx(cf0.corr['r'][llim:], (cf0.corr['corr']/cf_ill1.corr['corr'])[llim:], label='input', alpha=alpha, color=inn_c, linewidth=lw)
ax2.semilogx(cf1.corr['r'][llim:], (cf1.corr['corr']/cf_ill1.corr['corr'])[llim:], label='output', alpha=alpha, color=out_c, linewidth=lw)
ax2.semilogx(cf2.corr['r'][llim:], (cf2.corr['corr']/cf_ill1.corr['corr'])[llim:], label='target', alpha=alpha, color=tgt_c, linewidth=lw)
ax2.semilogx(cf_ill2.corr['r'][llim:], (cf_ill2.corr['corr']/cf_ill1.corr['corr'])[llim:], label='Illustris-3', ls='dashdot', alpha=alpha, color=ill3_c, linewidth=lw)
ax2.semilogx(cf_ill1.corr['r'][llim:], (cf_ill1.corr['corr']/cf_ill1.corr['corr'])[llim:], label='Illustris-2', ls='dashed', alpha=0.4, color=ill2_c, linewidth=lw)
plt.setp(ax1.get_xticklabels(), visible=False) #remove tick labels
ax2.tick_params(axis='x',labeltop=False, bottom=True, top=True, which='both')
ax2.tick_params(axis='y', which='both', left=True)
ax2.set_xlabel('r [Mpc/h]')
ax2.set_ylabel(r'$\dfrac{\zeta_2 (r)}{\zeta_{2, target}}$')

fig.subplots_adjust(hspace=0)
plt.savefig(direc_plots+'halo_2pcf_0.pdf')

##################################################################

lowlimit=bins[1]
highlimit=bins[2]

fig = plt.figure(figsize=(12*2,12))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])

mask_inn = (cat_mass_inn >= lowlimit) & (cat_mass_inn < highlimit)
masked_pos_inn = fof_inn['Position'][mask_inn]

mask_out = (cat_mass_out >= lowlimit) & (cat_mass_out < highlimit)
masked_pos_out = fof_out['Position'][mask_out]

mask_tgt = (cat_mass_tgt >= lowlimit) & (cat_mass_tgt < highlimit)
masked_pos_tgt = fof_tgt['Position'][mask_tgt]

data0 = {}
data0['Position'] = np.array(masked_pos_inn)
n_inn = data0['Position'].shape[0]
rand0 = {}
rand0['Position'] = np.random.uniform(0, box_sz, size=(n_inn*10,3))
data1 = {}
data1['Position'] = np.array(masked_pos_out)
n_out = data1['Position'].shape[0]
rand1 = {}
rand1['Position'] = np.random.uniform(0, box_sz, size=(n_out*10,3))
data2 = {}
data2['Position'] = np.array(masked_pos_tgt)
n_tgt = data2['Position'].shape[0]
rand2 = {}
rand2['Position'] = np.random.uniform(0, box_sz, size=(n_tgt*10,3))

print(data2['Position'].min(),data2['Position'].max())
print(rand2['Position'].min(),rand2['Position'].max())

cat0 = ArrayCatalog(data0, comm=comm)
cat_rand0 = ArrayCatalog(rand0, comm=comm)
cat1 = ArrayCatalog(data1, comm=comm)
cat_rand1 = ArrayCatalog(rand1, comm=comm)
cat2 = ArrayCatalog(data2, comm=comm)
cat_rand2 = ArrayCatalog(rand2, comm=comm)

#calc auto corr
auto_bins = np.logspace(-1, 1, 25) #in Mpc/h

cf0 = SimulationBox2PCF('1d', data1=cat0, randoms1=cat_rand0, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)
cf1 = SimulationBox2PCF('1d', data1=cat1, randoms1=cat_rand1, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)
cf2 = SimulationBox2PCF('1d', data1=cat2, randoms1=cat_rand2, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)

#illustris data catalog
#import Ill-3 groupcat and get halos
basePath2 = '/scratch/ds6311/Illustris-3/groupcat/'

h = 0.704 #from wmap9

halos2 = il.groupcat.loadHalos(basePath2,135)
halos2['Mass'] = halos2['GroupMass'] * 1e10 / h #halos['GroupMass'] in 10^10 𝑀⊙/ℎ
mass_mask2 = (halos2['Mass'] >= lowlimit) & (halos2['Mass'] < highlimit) & (halos2['GroupPos'][:,0] >= box_sz*1e3) & (halos2['GroupPos'][:,1] >= box_sz*1e3) & (halos2['GroupPos'][:,2] >= box_sz*1e3)

masked_pos2 = halos2['GroupPos'][mass_mask2,:] / 1e3 - box_sz #convert to Mpc/h

#import Ill-2 groupcat and get halos
basePath1 = '/scratch/ds6311/Illustris-2/groupcat/'

halos1 = il.groupcat.loadHalos(basePath1,135)
halos1['Mass'] = halos1['GroupMass'] * 1e10 / h #halos['GroupMass'] in 10^10 𝑀⊙/ℎ
mass_mask1 = (halos1['Mass'] >= lowlimit) & (halos1['Mass'] < highlimit) & (halos1['GroupPos'][:,0] >= box_sz*1e3) & (halos1['GroupPos'][:,1] >= box_sz*1e3) & (halos1['GroupPos'][:,2] >= box_sz*1e3)

masked_pos1 = halos1['GroupPos'][mass_mask1,:] / 1e3 - box_sz

data_ill2 = {}
data_ill2['Position'] = np.array(masked_pos2)
n_ill2 = data_ill2['Position'].shape[0]
rand_ill2 = {}
rand_ill2['Position'] = np.random.uniform(0, box_sz, size=(n_ill2*10,3))
data_ill1 = {}
data_ill1['Position'] = np.array(masked_pos1)
n_ill1 = data_ill1['Position'].shape[0]
rand_ill1 = {}
rand_ill1['Position'] = np.random.uniform(0, box_sz, size=(n_ill1*10,3))

print(data_ill1['Position'].min(),data_ill1['Position'].max())
print(rand_ill1['Position'].min(),rand_ill1['Position'].max())

cat_ill2 = ArrayCatalog(data_ill2, comm=comm)
cat_rand_ill2 = ArrayCatalog(rand_ill2, comm=comm)
cat_ill1 = ArrayCatalog(data_ill1, comm=comm)
cat_rand_ill1 = ArrayCatalog(rand_ill1, comm=comm)

cf_ill2 = SimulationBox2PCF('1d', data1=cat_ill2, randoms1=cat_rand_ill2, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)
cf_ill1 = SimulationBox2PCF('1d', data1=cat_ill1, randoms1=cat_rand_ill1, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)

alpha = 0.7
lw = 6

#plot
llim = 4

ax1 = plt.subplot(gs[0,0])
ax1.grid(True, alpha=0.3)
#ax1.set_title('Plot ('+str(i)+')')
ax1.loglog(cf0.corr['r'][llim:], cf0.corr['corr'][llim:], label='input', alpha=alpha, color=inn_c, linewidth=lw)
ax1.loglog(cf1.corr['r'][llim:], cf1.corr['corr'][llim:], label='output', alpha=alpha, color=out_c, linewidth=lw)
ax1.loglog(cf2.corr['r'][llim:], cf2.corr['corr'][llim:], label='target', alpha=alpha, color=tgt_c, linewidth=lw)
ax1.loglog(cf_ill2.corr['r'][llim:], cf_ill2.corr['corr'][llim:], label='Illustris-3', ls='dashdot', alpha=alpha, color=ill3_c, linewidth=lw)
ax1.loglog(cf_ill1.corr['r'][llim:], cf_ill1.corr['corr'][llim:], label='Illustris-2', ls='dashed', alpha=alpha, color=ill2_c, linewidth=lw)
ax1.legend()
ax1.set_ylabel(r'$\zeta_2 (r) \ \{$%.0e $\leq M_{\rm halo} <$%.0e $M_{\odot} \}$' % (lowlimit, highlimit), labelpad=10)
fig.add_subplot(ax1)
ax1.tick_params(axis='x', labeltop=False, bottom=False)
ax1.tick_params(axis='y', which='both', left=True)

ax2 = plt.subplot(gs[1,0], sharex=ax1)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.grid(True, alpha=0.3)
ax2.semilogx(cf0.corr['r'][llim:], (cf0.corr['corr']/cf_ill1.corr['corr'])[llim:], label='input', alpha=alpha, color=inn_c, linewidth=lw)
ax2.semilogx(cf1.corr['r'][llim:], (cf1.corr['corr']/cf_ill1.corr['corr'])[llim:], label='output', alpha=alpha, color=out_c, linewidth=lw)
ax2.semilogx(cf2.corr['r'][llim:], (cf2.corr['corr']/cf_ill1.corr['corr'])[llim:], label='target', alpha=alpha, color=tgt_c, linewidth=lw)
ax2.semilogx(cf_ill2.corr['r'][llim:], (cf_ill2.corr['corr']/cf_ill1.corr['corr'])[llim:], label='Illustris-3', ls='dashdot', alpha=alpha, color=ill3_c, linewidth=lw)
ax2.semilogx(cf_ill1.corr['r'][llim:], (cf_ill1.corr['corr']/cf_ill1.corr['corr'])[llim:], label='Illustris-2', ls='dashed', alpha=0.4, color=ill2_c, linewidth=lw)
plt.setp(ax1.get_xticklabels(), visible=False) #remove tick labels
ax2.tick_params(axis='x',labeltop=False, bottom=True, top=True, which='both')
ax2.tick_params(axis='y', which='both', left=True)
ax2.set_xlabel('r [Mpc/h]')
ax2.set_ylabel(r'$\dfrac{\zeta_2 (r)}{\zeta_{2, target}}$')
#fig.add_subplot(ax2)


###################

lowlimit=bins[-1]
highlimit=1e20

mask_inn = (cat_mass_inn >= lowlimit) & (cat_mass_inn < highlimit)
masked_pos_inn = fof_inn['Position'][mask_inn]

mask_out = (cat_mass_out >= lowlimit) & (cat_mass_out < highlimit)
masked_pos_out = fof_out['Position'][mask_out]

mask_tgt = (cat_mass_tgt >= lowlimit) & (cat_mass_tgt < highlimit)
masked_pos_tgt = fof_tgt['Position'][mask_tgt]

data0 = {}
data0['Position'] = np.array(masked_pos_inn)
n_inn = data0['Position'].shape[0]
rand0 = {}
rand0['Position'] = np.random.uniform(0, box_sz, size=(n_inn*10,3))
data1 = {}
data1['Position'] = np.array(masked_pos_out)
n_out = data1['Position'].shape[0]
rand1 = {}
rand1['Position'] = np.random.uniform(0, box_sz, size=(n_out*10,3))
data2 = {}
data2['Position'] = np.array(masked_pos_tgt)
n_tgt = data2['Position'].shape[0]
rand2 = {}
rand2['Position'] = np.random.uniform(0, box_sz, size=(n_tgt*10,3))

print(data2['Position'].min(),data2['Position'].max())
print(rand2['Position'].min(),rand2['Position'].max())

cat0 = ArrayCatalog(data0, comm=comm)
cat_rand0 = ArrayCatalog(rand0, comm=comm)
cat1 = ArrayCatalog(data1, comm=comm)
cat_rand1 = ArrayCatalog(rand1, comm=comm)
cat2 = ArrayCatalog(data2, comm=comm)
cat_rand2 = ArrayCatalog(rand2, comm=comm)

#calc auto corr
auto_bins = np.logspace(-1, 1, 25) #in Mpc/h

cf0 = SimulationBox2PCF('1d', data1=cat0, randoms1=cat_rand0, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)
cf1 = SimulationBox2PCF('1d', data1=cat1, randoms1=cat_rand1, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)
cf2 = SimulationBox2PCF('1d', data1=cat2, randoms1=cat_rand2, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)

#illustris data catalog
#import Ill-3 groupcat and get halos
basePath2 = '/scratch/ds6311/Illustris-3/groupcat/'

h = 0.704 #from wmap9

halos2 = il.groupcat.loadHalos(basePath2,135)
halos2['Mass'] = halos2['GroupMass'] * 1e10 / h #halos['GroupMass'] in 10^10 𝑀⊙/ℎ
mass_mask2 = (halos2['Mass'] >= lowlimit) & (halos2['Mass'] < highlimit) & (halos2['GroupPos'][:,0] >= box_sz*1e3) & (halos2['GroupPos'][:,1] >= box_sz*1e3) & (halos2['GroupPos'][:,2] >= box_sz*1e3)

masked_pos2 = halos2['GroupPos'][mass_mask2,:] / 1e3 - box_sz #convert to Mpc/h

#import Ill-2 groupcat and get halos
basePath1 = '/scratch/ds6311/Illustris-2/groupcat/'

halos1 = il.groupcat.loadHalos(basePath1,135)
halos1['Mass'] = halos1['GroupMass'] * 1e10 / h #halos['GroupMass'] in 10^10 𝑀⊙/ℎ
mass_mask1 = (halos1['Mass'] >= lowlimit) & (halos1['Mass'] < highlimit) & (halos1['GroupPos'][:,0] >= box_sz*1e3) & (halos1['GroupPos'][:,1] >= box_sz*1e3) & (halos1['GroupPos'][:,2] >= box_sz*1e3)

masked_pos1 = halos1['GroupPos'][mass_mask1,:] / 1e3 - box_sz

data_ill2 = {}
data_ill2['Position'] = np.array(masked_pos2)
n_ill2 = data_ill2['Position'].shape[0]
rand_ill2 = {}
rand_ill2['Position'] = np.random.uniform(0, box_sz, size=(n_ill2*10,3))
data_ill1 = {}
data_ill1['Position'] = np.array(masked_pos1)
n_ill1 = data_ill1['Position'].shape[0]
rand_ill1 = {}
rand_ill1['Position'] = np.random.uniform(0, box_sz, size=(n_ill1*10,3))

print(data_ill1['Position'].min(),data_ill1['Position'].max())
print(rand_ill1['Position'].min(),rand_ill1['Position'].max())

cat_ill2 = ArrayCatalog(data_ill2, comm=comm)
cat_rand_ill2 = ArrayCatalog(rand_ill2, comm=comm)
cat_ill1 = ArrayCatalog(data_ill1, comm=comm)
cat_rand_ill1 = ArrayCatalog(rand_ill1, comm=comm)

cf_ill2 = SimulationBox2PCF('1d', data1=cat_ill2, randoms1=cat_rand_ill2, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)
cf_ill1 = SimulationBox2PCF('1d', data1=cat_ill1, randoms1=cat_rand_ill1, edges=auto_bins, BoxSize=box_sz, periodic=False, show_progress=False)

alpha = 0.7
lw = 6

#plot
llim = 4

ax3 = plt.subplot(gs[0,1])
ax3.grid(True, alpha=0.3)
#ax1.set_title('Plot ('+str(i)+')')
ax3.loglog(cf0.corr['r'][llim:], cf0.corr['corr'][llim:], label='input', alpha=alpha, color=inn_c, linewidth=lw)
ax3.loglog(cf1.corr['r'][llim:], cf1.corr['corr'][llim:], label='output', alpha=alpha, color=out_c, linewidth=lw)
ax3.loglog(cf2.corr['r'][llim:], cf2.corr['corr'][llim:], label='target', alpha=alpha, color=tgt_c, linewidth=lw)
ax3.loglog(cf_ill2.corr['r'][llim:], cf_ill2.corr['corr'][llim:], label='Illustris-3', ls='dashdot', alpha=alpha, color=ill3_c, linewidth=lw)
ax3.loglog(cf_ill1.corr['r'][llim:], cf_ill1.corr['corr'][llim:], label='Illustris-2', ls='dashed', alpha=alpha, color=ill2_c, linewidth=lw)
ax3.legend()
ax3.set_ylabel(r'$\zeta_2 (r) \ \{$%.0e $\leq M_{\rm halo} <$%.0e $M_{\odot} \}$' % (lowlimit, highlimit), labelpad=10)
fig.add_subplot(ax3)
ax3.tick_params(axis='x', labeltop=False, bottom=False)
ax3.tick_params(axis='y', which='both', left=True)

ax4 = plt.subplot(gs[1,1], sharex=ax3)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.grid(True, alpha=0.3)
ax4.semilogx(cf0.corr['r'][llim:], (cf0.corr['corr']/cf_ill1.corr['corr'])[llim:], label='input', alpha=alpha, color=inn_c, linewidth=lw)
ax4.semilogx(cf1.corr['r'][llim:], (cf1.corr['corr']/cf_ill1.corr['corr'])[llim:], label='output', alpha=alpha, color=out_c, linewidth=lw)
ax4.semilogx(cf2.corr['r'][llim:], (cf2.corr['corr']/cf_ill1.corr['corr'])[llim:], label='target', alpha=alpha, color=tgt_c, linewidth=lw)
ax4.semilogx(cf_ill2.corr['r'][llim:], (cf_ill2.corr['corr']/cf_ill1.corr['corr'])[llim:], label='Illustris-3', ls='dashdot', alpha=alpha, color=ill3_c, linewidth=lw)
ax4.semilogx(cf_ill1.corr['r'][llim:], (cf_ill1.corr['corr']/cf_ill1.corr['corr'])[llim:], label='Illustris-2', ls='dashed', alpha=0.4, color=ill2_c, linewidth=lw)
plt.setp(ax3.get_xticklabels(), visible=False) #remove tick labels
ax4.tick_params(axis='x',labeltop=False, bottom=True, top=True, which='both')
ax4.tick_params(axis='y', which='both', left=True)
ax4.set_xlabel('r [Mpc/h]')
ax4.set_ylabel(r'$\dfrac{\zeta_2 (r)}{\zeta_{2, target}}$')

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0.3)

plt.savefig(direc_plots+'halo_2pcf_1.pdf')

