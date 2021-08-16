import h5py
import numpy as np
import os

import illustris_python as il
from nbodykit.source.catalog.array import ArrayCatalog
from argparse import ArgumentParser

def crops(fields, anchor, crop, pad, size):
    """
    Adapted from Yin Li's map2map code.
    """
    ndim = len(size)
    assert all(len(x) == ndim for x in [anchor, crop, pad, size]), 'inconsistent ndim'
    new_fields = []
    for x in fields: #loop over channel dim
        ind = []
        for d, (a, c, (p0, p1), s) in enumerate(zip(anchor, crop, pad, size)):
            i = np.arange(a - p0, a + c + p1)
            i %= s
            i = i.reshape((-1,) + (1,) * (ndim - d - 1))
            ind.append(i)
            
        x = x[tuple(ind)]

        new_fields.append(x)
    return np.array(new_fields)


#####################################

#make parser
parser = ArgumentParser(description='create training data')

parser.add_argument('--simsize', type=float, default=75., help='sim box size in Mpc/h')
parser.add_argument('--pixside', type=int, default=2048, help='number of voxels per dimension the sim gets divided into')
parser.add_argument('--pad', type=int, default=20, help='padding pixels at each side per dimension')
parser.add_argument('--crop', type=int, default=128, help='size of individual (unpadded) training cube')
parser.add_argument('--illustris', type=str, default='3', help='Illustris sim type: 1,2 or 3')
parser.add_argument('--window', type=str, default='cic', help='interpolation window to be used when creating cubes')

args = parser.parse_args()

simsize = args.simsize #Mpc/h
pixside = args.pixside
pad = args.pad
crop = args.crop

illustris = args.illustris
window = args.window
basePath='/scratch/ds6311/Illustris-'+illustris+'/dm_only' #path of simulation catalog

print('pixside:', pixside)
print('pad:', pad)
print('crop:', crop)
print('illustris:', illustris)
print('window:', window, flush=True)

####################################

out_data=''

if window == 'cic' or window == 'tsc':
    out_data += '/'+window
    out_data = '/scratch/ds6311/Illustris-'+illustris+'/dm_only_'+str(pixside)+'_counts_subcubes'+str(crop)+'_pad'+str(pad)+'/'+window+'/'

if not os.path.exists(out_data):
    os.makedirs(out_data)
    
maxsize = 1024 # maximum cubesize such that window interpoltation still works due to int not long_int being used in mpi4py
times = int(pixside/maxsize) #times^3 large cubes will be created and then at the end divided into training cubes

#load dm pos:
particles = il.snapshot.loadSubset(basePath,135,'dm',['Coordinates']) / 1e3 #in Mpc/h

#resample entire sim cube and save
def unmakeNbdyDensity(cub, n_part, cubesize):
    """
    unmake nbodykit density field with mean dens := 1
    """

    n_voxels = cubesize**3

    avg = n_part / n_voxels

    mult_factor = 1./avg #factor to mult particles with in order to get mean density = 1 (like part_mass)

    counts = cub / mult_factor
    return counts

extrap = 0 #extra pixels to pad in order to remove window artifacts

if window == 'cic': extrap = 1
if window == 'tsc': extrap = 2
    
extrappad = extrap + pad

cubecounter = 0

#params for cropping mechanism below
crop_start = pad
crop_stop = maxsize + pad

print_edges = False #for debugging

if times != 1:
    for i in range(times):
        for j in range(times):
            for k in range(times):

                edge = [False, False, False] #store if we have an x,y or z edge case

                if (i == 0) | (i == times-1): edge[0] = True
                if (j == 0) | (j == times-1): edge[1] = True
                if (k == 0) | (k == times-1): edge[2] = True
                
                #limits to cat the particle catalogs with:
                xlim = [simsize/pixside*((i*maxsize-extrappad)%pixside), simsize/pixside*(((i+1)*maxsize+extrappad)%pixside)] #in Mpc/h
                ylim = [simsize/pixside*((j*maxsize-extrappad)%pixside), simsize/pixside*(((j+1)*maxsize+extrappad)%pixside)]
                zlim = [simsize/pixside*((k*maxsize-extrappad)%pixside), simsize/pixside*(((k+1)*maxsize+extrappad)%pixside)]

                parts = []

                if not edge[0] and not edge[1] and not edge[2]:
                    parts = particles[((particles[:,0] >= xlim[0]) & (particles[:,0] < xlim[1])) & ((particles[:,1] >= ylim[0]) & (particles[:,1] < ylim[1])) & ((particles[:,2] >= zlim[0]) & (particles[:,2] < zlim[1]))]
                    if print_edges: print('no edge')
                elif edge[0] and not edge[1] and not edge[2]:
                    parts = particles[((particles[:,0] >= xlim[0]) | (particles[:,0] < xlim[1])) & ((particles[:,1] >= ylim[0]) & (particles[:,1] < ylim[1])) & ((particles[:,2] >= zlim[0]) & (particles[:,2] < zlim[1]))]
                    if print_edges: print('x edge')
                elif not edge[0] and edge[1] and not edge[2]:
                    parts = particles[((particles[:,0] >= xlim[0]) & (particles[:,0] < xlim[1])) & ((particles[:,1] >= ylim[0]) | (particles[:,1] < ylim[1])) & ((particles[:,2] >= zlim[0]) & (particles[:,2] < zlim[1]))]
                    if print_edges: print('y edge')
                elif not edge[0] and not edge[1] and edge[2]:
                    parts = particles[((particles[:,0] >= xlim[0]) & (particles[:,0] < xlim[1])) & ((particles[:,1] >= ylim[0]) & (particles[:,1] < ylim[1])) & ((particles[:,2] >= zlim[0]) | (particles[:,2] < zlim[1]))]
                    if print_edges: print('z edge')
                elif edge[0] and edge[1] and not edge[2]:
                    parts = particles[((particles[:,0] >= xlim[0]) | (particles[:,0] < xlim[1])) & ((particles[:,1] >= ylim[0]) | (particles[:,1] < ylim[1])) & ((particles[:,2] >= zlim[0]) & (particles[:,2] < zlim[1]))]
                    if print_edges: print('x,y edge')
                elif edge[0] and not edge[1] and edge[2]:
                    parts = particles[((particles[:,0] >= xlim[0]) | (particles[:,0] < xlim[1])) & ((particles[:,1] >= ylim[0]) & (particles[:,1] < ylim[1])) & ((particles[:,2] >= zlim[0]) | (particles[:,2] < zlim[1]))]
                    if print_edges: print('x,z edge')
                elif not edge[0] and edge[1] and edge[2]:
                    parts = particles[((particles[:,0] >= xlim[0]) & (particles[:,0] < xlim[1])) & ((particles[:,1] >= ylim[0]) | (particles[:,1] < ylim[1])) & ((particles[:,2] >= zlim[0]) | (particles[:,2] < zlim[1]))]
                    if print_edges: print('y,z edge')
                else:
                    parts = particles[((particles[:,0] >= xlim[0]) | (particles[:,0] < xlim[1])) & ((particles[:,1] >= ylim[0]) | (particles[:,1] < ylim[1])) & ((particles[:,2] >= zlim[0]) | (particles[:,2] < zlim[1]))]
                    if print_edges: print('x,y,z edge')


                #shift particles to origin subcube:
                parts[:,0] -= (i*maxsize)/pixside * simsize
                parts[:,1] -= (j*maxsize)/pixside * simsize
                parts[:,2] -= (k*maxsize)/pixside * simsize

                #shift pos of periodic particles to origin and move the rest accordingly
                #later use of window requires this
                parts += (extrappad%pixside)/pixside * simsize
                parts = parts % simsize

                n_part = parts.shape[0]
                data = np.zeros(n_part, dtype=[('Position', ('f8', 3))])
                data['Position'] = parts
                cat = ArrayCatalog(data)
                cat.attrs['Size'] = float(n_part)
                cat.attrs['BoxSize'] = (simsize/pixside)*(maxsize+2*extrappad)

                mesh = cat.to_mesh(Nmesh=maxsize+2*extrappad, resampler=window, interlaced=False, compensated=True) #+2 to add 2 pixels per at edges to introduce no periodic artifacts while window sampling

                np_arr = unmakeNbdyDensity(mesh.preview(), n_part, maxsize+2*extrappad)

                field = np.zeros((1,maxsize+int(2*pad),maxsize+int(2*pad),maxsize+int(2*pad)))

                field[0] = np_arr[extrap:-extrap,extrap:-extrap,extrap:-extrap] #remove the window padded pixels


                size = field.shape[1:]
                size = np.asarray(size)
                ndim = len(size)
                
                #divide large cube into individual training cubes:

                newcrop = np.broadcast_to(crop, (ndim,))

                crop_start = np.broadcast_to(crop_start, (ndim,))
                crop_stop = np.broadcast_to(crop_stop, (ndim,))
                crop_step = newcrop

                anchors = np.stack(np.mgrid[tuple(
                    slice(crop_start[d], crop_stop[d], crop_step[d])
                    for d in range(ndim)
                    )], axis=-1).reshape(-1, ndim)

                ncrop = len(anchors)
                padd = np.broadcast_to(pad, (ndim, 2))

                for anchor in anchors:
                    cropped = crops(field, anchor, newcrop, padd, size)

                    dirname = 'cube' + str(cubecounter) + '_' + str(i*maxsize) + '_' + str(j*maxsize) + '_' + str(k*maxsize)
                    savepath = out_data+dirname
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)

                    np.save(savepath+'/cube_'+str(anchor[0]-pad)+'_'+str(anchor[1]-pad)+'_'+str(anchor[2]-pad)+'.npy', cropped)

                cubecounter += 1

else: #case where pixelsize is large enough st. all training cubes fit into 1 maxsize cube (=> no edge cases)
    
    parts = particles
    
    #shift pos of negative periodic particles to origin and move the rest accordingly (in positive direction)
    #later use of window requires this
    parts += (extrappad%pixside)/pixside * simsize
    parts = parts % simsize

    n_part = parts.shape[0]
    data = np.zeros(n_part, dtype=[('Position', ('f8', 3))])
    data['Position'] = parts
    cat = ArrayCatalog(data)
    cat.attrs['Size'] = float(n_part)
    cat.attrs['BoxSize'] = (simsize/pixside)*(maxsize+2*extrappad)

    mesh = cat.to_mesh(Nmesh=maxsize+2*extrappad, resampler=window, interlaced=False, compensated=True) #+2 to add 2 pixels per at edges to introduce no periodic artifacts while window sampling

    np_arr = unmakeNbdyDensity(mesh.preview(), n_part, maxsize+2*extrappad)

    field = np.zeros((1,maxsize+int(2*pad),maxsize+int(2*pad),maxsize+int(2*pad))) #maxsize+int(2*pad)

    field[0] = np_arr[extrap:-extrap,extrap:-extrap,extrap:-extrap] #remove the window padded pixels


    size = field.shape[1:]
    size = np.asarray(size)
    ndim = len(size)

    newcrop = np.broadcast_to(crop, (ndim,))

    crop_start = np.broadcast_to(crop_start, (ndim,))
    crop_stop = np.broadcast_to(crop_stop, (ndim,))
    crop_step = newcrop

    anchors = np.stack(np.mgrid[tuple(
        slice(crop_start[d], crop_stop[d], crop_step[d])
        for d in range(ndim)
        )], axis=-1).reshape(-1, ndim)

    ncrop = len(anchors)
    padd = np.broadcast_to(pad, (ndim, 2))

    for anchor in anchors:
        cropped = crops(field, anchor, newcrop, padd, size)

        savepath = out_data
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        np.save(savepath+'/cube_'+str(anchor[0]-pad)+'_'+str(anchor[1]-pad)+'_'+str(anchor[2]-pad)+'.npy', cropped)

    cubecounter += 1

