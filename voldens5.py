#import orionfits as of
import scipy.ndimage.measurements as imm
from scipy.interpolate import interp1d
import numpy as np
from os import mkdir
try:
    import astropy.io.fits as pf
except ImportError:
    import pyfits as pf

#TODO: memory !!!!!
#TODO: elongation: J-plots ? skeletonization (binary opening/closing) ?
#TODO: fragmentation along the line of sight: "fragmentedness": dendrograms?
#Fragmented Anisotropic Elongated Disconnected Volume Density Statistical Estimation Inference Tool Code Algorithm


pc_to_cm = 3.085677581e18

def flattening(coldens):
  """To be run first to get a sense of the size and complexity of the dataset.
  The complexity of the data is defined as the number of unique values divided
  by the number of data points.
  
  Parameters
  -----
  coldens: 2darray
    The column density data, can be masked (in some unit per cm-2)
  
  Returns
  -----
  u: 1darray
    All the unique values present in coldens, sorted
  nu: int
    The number of unique values in coldens""" 
  if isinstance(coldens,np.ma.masked_array):
    coldens = coldens.compressed()
  u = np.unique(coldens)
  coldens = coldens.flatten()
  s = coldens.shape[0]
  #u = u[u>0]
  nu = len(u)
  print(str(nu)+' unique values')
  print(str(int(100*nu/s))+'% of complexity')
  return u,nu

def compressing(coldens,nsamples = 1e6, method='linmm'):
  """Reduces the dataset complexity by binning the column density values. The
  returned values are bin lower boundaries.
  
  Parameters
  -----
  coldens: 2darray
    The column density data, can be masked (in some unit per cm-2)
  
  Options
  -----
  nsamples: int
    Number of sample values (default: 1e6)
  method: str, of callable function
    Sampling method, setting the bin spacing. The samples can be set by the
    minimum and maximum of the data, with linear, logarithmic or inverse
    logarithmic spacing ("linmm", "logmm", or "ilogmm"), or defined by linearly,
    logarithmically or inverse-logarithmically spaced percentiles ("linperc",
    "logperc", or "ilogperc"). Logarithmic spacing samples better the low
    column density values, inverse logarithmic the high ones. Percentile
    sampling follows better the structure of the data and avoids empty bins.
    Example: in a dataset with an approximately logarithmic PDF, the "ilogperc"
    method will return bins which will be roughly linearly spaced.
    One can also provide a callable method taking 'coldens' and 'nsamples' as
    arguments and returning the proper number of column density values
    (default: "linmm")
  
  Returns
  -----
  compressed: 1darray
    Sorted and sampled column density values
  """ 
  u,lu = flattening(coldens)
  if lu <=1:
    raise ValueError('The column density field is flat.')
  nsamples = int(nsamples)
  if nsamples < 1:
    nsamples = 1e6
  if nsamples > lu:
    nsamples = lu
  if nsamples == lu:
    compressed = u
  else:
    if method not in ['linmm','logmm','ilogmm','linperc','logperc','ilogperc']:
      if not callable(method):
        method = 'lin'
        print('WARNING!\n','Provided method is not callable, defaulting to linear sampling.')
    if method == 'linmm':
      p = np.linspace(u.min(),u.max(),nsamples+1)[:-1]
      f = interp1d(u,u,bounds_error=False,fill_value="extrapolate")
      compressed = f(p)
    elif method == 'logmm':
      p = np.logspace(np.log10(u.min()),np.log10(u.max()),nsamples+1)[:-1]
      f = interp1d(u,u,bounds_error=False,fill_value="extrapolate")
      compressed = f(p)
    elif method == 'ilogmm':
      l = np.log10(np.linspace(1,100,nsamples+1))/2
      p = u.min() + (u.max()-u.min())*l[:-1]
      f = interp1d(u,u,bounds_error=False,fill_value="extrapolate")
      compressed = f(p)
    elif method == 'linperc':
      p = np.linspace(0,100,nsamples+1)
      compressed = np.percentile(u,p[:-1],interpolation='nearest')
    elif method == 'logperc':
      start = 100/nsamples #first samples matched to linperc
      p = np.logspace(np.log10(start),2,nsamples+1)
      p[0] = 0 #include the minimum in the first bin
      compressed = np.percentile(u,p[:-1],interpolation='nearest')
    elif method == 'ilogperc':
      p = np.log10(np.linspace(1,100,nsamples+1))*50
      compressed = np.percentile(u,p[:-1],interpolation='nearest')
    else:
      compressed = method(u,nsamples)
      if len(compressed) != nsamples:
        print('Warning! The provided method does not return the right number of samples.')
  return compressed

def coldens_incrementing(coldens, unicoldens):
  """Gets the average values of the dataset in each bin, and ensures that no
  bin is empty.
  
  Parameters
  -----
  coldens: 2darray
    The column density data, can be masked (in some unit per cm-2)
  unicoldens: 1darray
    Flattened, sorted (and possibly compressed) version of 'coldens', which
    provides (lower) boundaries for binning the data
  
  Returns
  -----
  coldens_increments: 1darray
    Mean value of the column density data in each bin
  unicoldens_c: 1darray
    Copy of 'unicoldens' cleaned of all the values which led to empty bins.
    Each value 'coldens_increments[i]' therefore corresponds to the average of
    all 'coldens' values between 'unicoldens_c[i]' and 'unicoldens_c[i+1]'
  """
  l = len(unicoldens)
  binned_coldens = np.zeros(unicoldens.shape)
  k = int(max(10.**(int(np.log10(l-1))-2),1.))
  for i in range(l-1):
    if i%k == 0:
      print('\r\r\r{}%'.format(int(i/l*100.)),end='',flush=True) 
    binned_coldens[i] = np.ma.masked_array(coldens,mask = np.logical_or(coldens<unicoldens[i],coldens>=unicoldens[i+1])).mean()
    if np.isnan(binned_coldens[i]):
      if i == 0:
        binned_coldens[i] = 0
  print('\r\r\r100%',end='\n',flush=True)
  binned_coldens[-1] = np.ma.masked_array(coldens,mask = coldens<unicoldens[-1]).mean()
  unicoldens_c = unicoldens[~np.isnan(binned_coldens)]
  binned_coldens = binned_coldens[~np.isnan(binned_coldens)]
  bcd1 = np.array([0]+list(binned_coldens)[:-1])
  coldens_increments = binned_coldens-bcd1
  print(str(len(unicoldens) - len(unicoldens_c))+' empty bins')
  return coldens_increments, unicoldens_c

def connected_voldens(coldens, unicoldens, coldens_inc, scale): #TODO: adapt from disconnected
  if l == 0:
    l = len(unicoldens)
  voldens_peak = np.zeros(coldens.shape,dtype=np.float32)
  volumes = np.zeros((coldens.shape[0],coldens.shape[1],l),dtype=np.float32)
  voldensities = np.zeros((coldens.shape[0],coldens.shape[1],l),dtype=np.float32)
  k = int(max(10.**(int(np.log10(l-1))-2),1.))
  for i in range(l):
    if i%k == 0:
      print('\r\r\r{}%'.format(int(i/l*100)),end='',flush=True)
    O = np.zeros(coldens.shape)
    O[coldens>=compressed_coldens[i]] = 1.
    surf = O.sum()
    depths = O*np.sqrt(surf)*scale # *correction factor TODO
    depthsm1 = O/(np.sqrt(surf)*scale) # *correction factor TODO
    volumes[:,:,i] = depths*scale**2      
    voldensities[:,:,i] = coldens_increments[i]*depthsm1
    voldens_peak += coldens_increments[i]*depthsm1
  print('\r\r\r100%',end='\n',flush=True)
  
  print('\nPost-processing...')
  volumes2 = np.empty(coldens.shape,dtype=object)
  voldensities2 = np.empty(coldens.shape,dtype=object)
  for i in range(coldens.shape[0]):
    print('\r\r\r{}%'.format(int(i/coldens.shape[0]*100)),end='',flush=True)
    for j in range(coldens.shape[1]):
      v = volumes[i,j][volumes[i,j]>0] #removing empty increments
      v1 = v.copy()
      v1[:-1] -= v[1:] #nesting the volumes
      vd = voldensities[i,j][voldensities[i,j]>0] #removing empty increments
      vd = np.cumsum(vd) #increments to densities
      
      vd = vd[v1 > 0] #removing empty volumes
      v1 = v1[v1 > 0] #removing empty volumes
      
      v1 = list(v1)
      vd = list(vd)
      volumes2[i,j] = v1
      voldensities2[i,j] = vd
      volumes2flat.extend(v1) #
      voldensities2flat.extend(vd) #
  print('\r\r\r100%','\nFinishing...',end='\n',flush=True)
  
  voldens_pdf_map = (voldensities2,volumes2)
  #volumes2flat = list(volumes2.flatten())
  #voldensities2flat = list(voldensities2.flatten())
  #voldens_pdf_flat = (np.array([x for y in voldensities2flat for x in y]),np.array([x for y in volumes2flat for x in y]))
  voldens_pdf_flat = (np.array(voldensities2flat),np.array(volumes2flat)) #
  return voldens_peak, voldens_pdf_map, voldens_pdf_flat

def disconnected_voldens0(coldens, unicoldens, coldens_increments, scale): #DEPRECATED
  """
  Parameters
  -----
  coldens: 2darray
    The column density data, can be masked (in some unit per cm-2)
  unicoldens: 1darray
    Flattened, sorted (and possibly compressed) version of 'coldens'
  coldens_increments: 1darray
    Mean value of the data in the bins defined by 'unicoldens'
  scale: float
    Conversion factor to obtain the physical scale of pixels in cm
  
  Returns
  -----
  stuff
  """
  l = len(unicoldens)
  voldens_peak = np.zeros(coldens.shape,dtype=np.float32)
  volumes = np.zeros((coldens.shape[0],coldens.shape[1],l),dtype=np.float32)
  voldens_increments = np.zeros((coldens.shape[0],coldens.shape[1],l),dtype=np.float32)
  k = int(max(10.**(int(np.log10(l-1))-2),1.))
  for i in range(l):
    if i%k == 0:
      print('\r\r\r{}%'.format(int(i/l*100)),end='',flush=True)
    O = np.zeros(coldens.shape)
    O[coldens>=unicoldens[i]] = 1.
    region,nreg = imm.label(O)
    surf = imm.sum(O,region,index=range(1,nreg+1))
    depths = O.copy()
    depthsm1 = O.copy() #depths^-1
    areas = O.copy()
    for r in range(nreg):
      depths[region==r+1] = np.sqrt(surf[r])*scale# *correction factor TODO
      depthsm1[region==r+1] = 1/(np.sqrt(surf[r])*scale)# *correction factor TODO
    volumes[:,:,i] = depths*scale**2/(3.085677581e18**3)      #volumes in parsecs
    voldens_increments[:,:,i] = coldens_increments[i]*depthsm1
    voldens_peak += coldens_increments[i]*depthsm1
  print('\r\r\r100%',end='\n',flush=True)
  
  print('\nPost-processing...')
  volumes2 = np.empty(coldens.shape,dtype=object)
  voldensities2 = np.empty(coldens.shape,dtype=object)
  volumes2flat = [] #
  voldensities2flat = [] #
  for i in range(coldens.shape[0]):
    print('\r\r\r{}%'.format(int(i/coldens.shape[0]*100)),end='',flush=True)
    for j in range(coldens.shape[1]):
      vlist = list(volumes[i,j,1:])
      if 0 in vlist:
        i0 = vlist.index(0)
        if i0+1<len(vlist):
          if vlist[i0+1] > 0:
            print(i,j,i0,vlist[i0+1])
      v = volumes[i,j][volumes[i,j]>0] #removing empty increments
      v1 = v.copy()
      v1[:-1] -= v[1:] #nesting the volumes
      vd = voldens_increments[i,j][voldens_increments[i,j]>0] #removing empty increments
      if voldens_increments[i,j,0]==0: #in case of zero background
        vd = np.array([0]+list(vd))
      vd = np.cumsum(vd) #increments to densities
      
      vd = vd[v1 > 0] #removing empty volumes
      v1 = v1[v1 > 0] #removing empty volumes
      
      v1 = list(v1)
      vd = list(vd)
      volumes2[i,j] = v1
      voldensities2[i,j] = vd
      volumes2flat.extend(v1) #
      voldensities2flat.extend(vd) #
  print('\r\r\r100%','\nFinishing...',end='\n',flush=True)
  
  voldens_pdf_map = (voldensities2,volumes2)
  #volumes2flat = list(volumes2.flatten())
  #voldensities2flat = list(voldensities2.flatten())
  #voldens_pdf_flat = (np.array([x for y in voldensities2flat for x in y]),np.array([x for y in volumes2flat for x in y]))
  voldens_pdf_flat = (np.array(voldensities2flat),np.array(volumes2flat)) #
  return voldens_peak, voldens_pdf_map, voldens_pdf_flat

def disconnected_voldens(coldens, unicoldens, coldens_increments, scale):
  """
  Parameters
  -----
  coldens: 2darray
    The column density data, can be masked (in some unit per cm-2)
  unicoldens: 1darray
    Flattened, sorted (and possibly compressed) version of 'coldens'
  coldens_increments: 1darray
    Mean value of the data in the bins defined by 'unicoldens'
  scale: float
    Conversion factor to obtain the physical scale of pixels in cm
  
  Returns
  -----
  voldensities: 3darray
    The inferred volume densities present along each line of sight, sorted in
    increasing order (in some unit per cm-3, depending on 'coldens'). The last
    dimension of the array (depth) is equal to the length of 'unicoldens', and
    corresponds to most complex (densest) line of sight; most lines of sights
    contain a lot of zeros after their highest volume density has been reached
  volumes: 3darray
    The volume (in pc^3, i.e. physical area of one pixel times inferred depth)
    correpsonding to the volume densities. Note that 'volumes[x,y,z]' is the
    volume AT a volume density 'voldensities[x,y,z]', not the volume ABOVE this
    value.
  
  Examples
  -----
  c_coldens = compressing(coldens)
  coldens_inc,c_coldens2 = coldens_incrementing(coldens,c_coldens)
  scale = 5e16
  vol_dens,vol = disconnected_voldens(coldens, c_coldens2, coldens_inc, scale)
  Vtot = vol.sum()                          # total volume of the cloud, in pc3
  Mtot = (vol_dens*vol).sum()*pc_to_cm**3   # total mass of the cloud, in the
                                            #  unit of 'coldens'*cm2
  vpeak = vol_dens.max(-1)                  # highest volume density reached
                                            #  along the l o s
  """
  l = len(unicoldens)
  #voldens_peak = np.zeros(coldens.shape,dtype=np.float32)
  volumes = np.zeros((coldens.shape[0],coldens.shape[1],l),dtype=np.float32)
  voldens_increments = np.zeros((coldens.shape[0],coldens.shape[1],l),dtype=np.float32)
  k = int(max(10.**(int(np.log10(l-1))-2),1.))
  for i in range(l):
    if i%k == 0:
      print('\r\r\r{}%'.format(int(i/l*100)),end='',flush=True)
    O = np.zeros(coldens.shape)
    O[coldens>=unicoldens[i]] = 1.
    region,nreg = imm.label(O)
    surf = imm.sum(O,region,index=range(1,nreg+1))
    depths = O.copy()
    depthsm1 = O.copy() #depths^-1
    areas = O.copy()
    for r in range(nreg):
      depths[region==r+1] = np.sqrt(surf[r])*scale# *correction factor TODO
      depthsm1[region==r+1] = 1/(np.sqrt(surf[r])*scale)# *correction factor TODO
    volumes[:,:,i] = depths*scale**2/(pc_to_cm**3)      #volumes in parsecs
    voldens_increments[:,:,i] = coldens_increments[i]*depthsm1
    #voldens_peak += coldens_increments[i]*depthsm1
  print('\r\r\r100%',end='\n',flush=True)
  
  print('\nPost-processing...')
  volumes2 = volumes.copy()
  volumes2[:,:,:-1] -= volumes[:,:,1:]
  voldensities = np.cumsum(voldens_increments,axis=-1)
  voldensities[voldens_increments == 0] = 0
  return voldensities, volumes2

def connected_fractal_voldens(): #TODO
  raise NotImplementedError("Working on it!")

def disconnected_fractal_voldens(): #TODO
    raise NotImplementedError("Working on it!")

def voldens(coldens, nsamples=1e4, compression='linmm', pxlres=9, distance=400, connected=False, simple_depth=True, fractal=False):
  """Main framework for volume density estimation.
  
  Parameters
  -----
  coldens: 2darray
    The column density data, can be masked (in some unit per cm-2)
  
  Options
  -----
  nsamples: int
    Number of sample values (default: 1e6)
  compression: str, of callable function
    Sampling method. Can be linear, logarithmic or inverse logarithmic, set by
    the min and max of the data or its percentiles ("linmm", "logmm", "ilogmm",
    "linperc", "logperc", "ilogperc"). One can also provide a callable method.
    See 'compressing' for details (default: "linmm")
  pxlres: float
    Spatial resolution of 'coldens', in arsec per pixel (default: 9, ORION-B)
  distance: float
    Distance to the studied object, in parsecs (default: 400, Orion B)
  connected: bool
    If True, (default: False)
  simple_depth: bool /// WARNING, NOT IMPLEMENTED, ideally should be False
    (default: True) 
  fractal: bool /// WARNING, NOT IMPLEMENTED, ideally should be True (?)
    (default: False)

  Returns
  -----
  voldensities: 3darray
    The inferred volume densities present along each line of sight, sorted in
    increasing order (in some unit per cm-3, depending on 'coldens'). The last
    dimension of the array (depth) is equal to the length of 'unicoldens', and
    corresponds to most complex (densest) line of sight; most lines of sights
    contain a lot of zeros after their highest volume density has been reached
  volumes: 3darray
    The volume (in pc^3, i.e. physical area of one pixel times inferred depth)
    correpsonding to the volume densities. Note that 'volumes[x,y,z]' is the
    volume AT a volume density 'voldensities[x,y,z]', not the volume ABOVE this
    value.
  
  Examples
  -----
  vol_dens,vol = voldens(coldens, nsamples=1e4, distance=230, connected=True)
  Vtot = vol.sum()                          # total volume of the cloud, in pc3
  Mtot = (vol_dens*vol).sum()*pc_to_cm**3   # total mass of the cloud, in the
                                            #  unit of 'coldens'*cm2
  vpeak = vol_dens.max(-1)                  # highest volume density reached
                                            #  along the l o s
  """
  scale = pxlres*distance*1.495978707e13 #pixel physical scale in cm
  print('Compressing...')
  coldens = np.ma.masked_array(coldens, mask = coldens <= 0)
  compressed_coldens = compressing(coldens, nsamples=nsamples, method=compression)
  coldens = coldens.filled(0)
  l = len(compressed_coldens)
  
  print('\nExtracting column density increments...')
  coldens_increments,clean_compressed_coldens = coldens_incrementing(coldens, compressed_coldens)
  
  print('\nAnalyzing volumes...')
  if connected:
    if fractal:
      print('(connected, fractal computation)')
      #compute fdim2d of contours
      #obtain fdim3d and volume
      #add voldens increment
      pass
    else:
      print('(connected, simple area-to-volume computation)')
      voldensities, volumes = connected_voldens(coldens, clean_compressed_coldens, coldens_increments, scale)
      
  else:
    if fractal:
      print('(disconnected, fractal computation)')
      #identify regions
      #compute fdim2d of each region
      #obtain fdim3d and volume
      #add voldens increments
      pass
    else:
      print('(disconnected, simple area-to-volume computation)')
      voldensities, volumes = disconnected_voldens(coldens, clean_compressed_coldens, coldens_increments, scale)
      
  return voldensities, volumes

def save_all(voldensities,volumes,name,header=None):
  """Writes the volume density structure to the disk. Each row of lines of
  sight is saved as a separate NumPy file, to avoid having a single file of
  unmanageable weight. All the files are created in a new folder, together with
  a "shape.txt" file which records the dimension of the input cubes. Density
  and volume data are written to separate files. For a shape (N,M) of the
  field, a total of 2*N+1 files are written (or 2*N+2 with the header).
  
  Parameters
  -----
  voldensities: 3darray
    Volume densities found along each line of sight (in some unit per cm-3)
  volumes: 3darray
    Volumes corresponding to each volume density (in pc3)
  name: str
    Folder name. The volume density data will be save to ./name/dens-i.npy and
    the volume data to ./name/vol-i.npy, where 'i' is the number of the row. 
  
  Options
  -----
  header: pyfits.Header object, or str
    Astronomical description of the column density data used to compute the
    volume density, written to a ./name/header.txt file (default: None)
  """
  mkdir(name)
  if header is not None:
    if type(header) == str:
      f = open(name+'/header.txt',mode="w")
      f.write(header)
      f.close()
    else:
      header.totextfile(name+'/header.txt')
  s = voldensities.shape
  if s != volumes.shape:
    raise ValueError('Array sizes to not match.')
  f = open(name+'/shape.txt',mode="w")
  f.write(str(s))
  f.close()
  for i in range(voldensities.shape[0]):
    print('\r\r\r{}%'.format(int(i/voldensities.shape[0]*100)),end='',flush=True)
    av = volumes[i,:,:].copy()
    ad = voldensities[i,:,:].copy()
    np.save(name+'/vol-{}'.format(i), av)
    np.save(name+'/dens-{}'.format(i), ad)
  print('\r\r\r100%')

def save_simple(voldensities,volumes,name,header): #doc TODO
  peak = voldensities.max(-1)
  voldensitiesflat,volumesflat = cube_to_pdf(voldensities,volumes)
  head = header.copy()
  if 'DATAMIN' in head.keys():
    head['DATAMIN'] = (peak.min(),head.comments['DATAMIN'])
  if 'DATAMAX' in head.keys():
    head['DATAMAX'] = (peak.max(),head.comments['DATAMAX'])
  if 'BUNIT' in head.keys():
    head['BUNIT'] = ('cm-3',head.comments['BUNIT'])
  pf.writeto(name+'-peak.fits',peak,head) #fix it
  f = open(name+'-pdf.txt',mode="w")
  f.write('Density\tVolume\n')
  f.write('cm-3\tpc3\n\n')
  for i in range(len(voldensitiesflat)):
    l = '{:.8e}\t{:.8e}\n'.format(voldensitiesflat[i],volumesflat[i])
    f.write(l)
  f.close()

def save_stat_map(voldensities,volumes,name,header,stat="peak",w="vol",p=50): #doc TODO
  """ peak, mean, std, perc """
  if w == "vol":
    wei = volumes
  elif w == "mass":
    wei = volumes*voldensities
  else:
    raise ValueError('Weighting must be by volume (w="vol") or by mass (w="mass").')

  if stat == "peak":
    statmap = voldensities.max(-1)
  elif stat == "mean":
    statmap = np.average(voldensities,axis=-1,weights=wei)
  elif stat == "std":
    mm = np.average(voldensities,axis=-1,weights=wei)
    statmap = np.sqrt(np.average((voldensities-mm[:,:,None])**2,axis=-1,weights=wei))
  elif stat == "perc":
    statmap = weighted_percentile(voldensities,q = p, weights=wei, axis=-1)
  else:
    raise ValueError('Choose a stat among "peak", "mean", "std" or "perc".')

  head = header.copy()
  if 'DATAMIN' in head.keys():
    head['DATAMIN'] = (statmap.min(),head.comments['DATAMIN'])
  if 'DATAMAX' in head.keys():
    head['DATAMAX'] = (statmap.max(),head.comments['DATAMAX'])
  if 'BUNIT' in head.keys():
    head['BUNIT'] = ('cm-3',head.comments['BUNIT'])
  if stat == "perc":
    pf.writeto(name+'-perc-'+str(p)+'.fits',statmap,head) #fix it TODO
  else:
    pf.writeto(name+'-'+stat+'.fits',statmap,head) #fix it TODO



def cube_to_pdf0(voldensities,volumes): #doc TODO - DEPRECATED ?
  print('Extracting the PDF...')
  vd = np.unique(voldensities.flatten())
  z = np.zeros(vd.shape)
  vdict = dict(zip(vd,z))
  vdf = voldensities.flatten()
  volf = volumes.flatten()
  l = len(vdf)
  k = int(max(10.**(int(np.log10(l-1))-2),1.))
  for i in range(l):
    if i%k == 0:
      print('\r\r\r{}%'.format(int(i/l*100.)),end='',flush=True)
    vdict[vdf[i]] += volf[i]
  print('\r\r\r100%',end='\n\n',flush=True)
  if 0. in vdict.keys():
    vdict.pop(0.)
  z = zip(vdict.keys(), vdict.values())
  zz = list(zip(*sorted(z)))
  voldensflat = np.array(zz[0])
  volflat = np.array(zz[1])
  return voldensflat,volflat


def cube_to_pdf(voldensities,volumes,save=False,name=None): #doc TODO
  print('Extracting the PDF...')
  vd = np.unique(voldensities.flatten())
  z = np.zeros(vd.shape)
  vdict = dict(zip(vd,z))
  vdf = voldensities.flatten()
  volf = volumes.flatten()
  l = len(vdf)
  k = int(max(10.**(int(np.log10(l-1))-2),1.))
  for i in range(l):
    if i%k == 0:
      print('\r\r\r{}%'.format(int(i/l*100.)),end='',flush=True)
    vdict[vdf[i]] += volf[i]
  print('\r\r\r100%',end='\n\n',flush=True)
  if 0. in vdict.keys():
    vdict.pop(0.)
  z = zip(vdict.keys(), vdict.values())
  zz = list(zip(*sorted(z)))
  voldensflat = np.array(zz[0])
  volflat = np.array(zz[1])
  if save:
    if type(name) == str:
      name = name+"-pdf.txt"
    else:
      name = "pdf.txt"
    f = open(name,mode="w")
    f.write('Density\tVolume\n')
    f.write('cm-3\tpc3\n\n')
    for i in range(len(voldensflat)):
      l = '{:.8e}\t{:.8e}\n'.format(voldensflat[i],volflat[i])
      f.write(l)
    f.close()
  return voldensflat,volflat



def cube_to_pdf_alt(voldensities,volumes,save=False,name=None): #doc TODO
  print('Extracting the PDF...')
  voldensflat,ind = np.unique(voldensities.flatten(),return_inverse=True)
  volflat = np.zeros(voldensflat.shape)
  vcopy = volumes.flatten()
  l = len(vcopy)
  k = int(max(10.**(int(np.log10(l-1))-2),1.))
  for i in range(l):
    if i%k == 0:
      print('\r\r\r{}%'.format(int(i/l*100.)),end='',flush=True)
    volflat[ind[i]] += vcopy[i]
  print('\r\r\r100%',end='\n\n',flush=True)
  if voldensflat[0] == 0:
    voldensflat = voldensflat[1:]
    volflat = volflat[1:]
  if save:
    if type(name) == str:
      name = name+"-pdf.txt"
    else:
      name = "pdf.txt"
    f = open(name,mode="w")
    f.write('Density\tVolume\n')
    f.write('cm-3\tpc3\n\n')
    for i in range(len(voldensflat)):
      l = '{:.8e}\t{:.8e}\n'.format(voldensflat[i],volflat[i])
      f.write(l)
    f.close()
  return voldensflat,volflat



def weighted_percentile_nope(data, q, weights=None, axis=None): #adapted from weighted_stats by Jack Peterson, MIT
  """Calculate the weighted median of an array/list using numpy."""
  if weights is None:
    return np.percentile(data,q,axis=axis)
  if q == 0:
    return np.min(data,axis=axis)
  if q == 100:
    return np.max(data,axis=axis)

  if q<0 or q>100:
    raise ValueError("The percentile needs to be between 0 and 100")
  if q<1:
    print("WARNING: the percentile needs to be between 0 and 100")
  if data.shape != weights.shape:
    raise ValueError("Data and weights must have the same dimensions.")
  if np.any(np.sum(weights, axis = axis) <= 0):
    raise ValueError('Weights should have a non-zero sum.')


  if type(axis) == int: #converting to a computation over axis 0 in 2 dimensions
    if axis != 0:
      weights1 = np.swapaxes(weights,axis,0)
      data1 = np.swapaxes(data,axis,0)
    else:
      weights1 = weights.copy()
      data1 = data.copy()
    s = weights1.shape
  else:
    weights1 = weights.flatten()
    data1 = data.flatten()
    s = weights1.shape
  #if axis != -1 and axis != weights.ndim -1:  #avoid an unnecessary swap of axes is possible
  weights1 = np.reshape(weights1,(s[0],int(np.prod(s)/s[0]))) #dimensions = stat x pseudospatial
  data1 = np.reshape(data1,(s[0],int(np.prod(s)/s[0]))) #dimensions = stat x pseudospatial

  #print(s)

  sorted_data, sorted_weights = parallel_sort(data1,weights1,axis=0)

  qpoint = q/100. * np.sum(sorted_weights, axis = 0) # q > 0 #dimensions = pseudospatial

  cumul_weight = np.cumsum(sorted_weights, axis=0) #dimensions = stat x pseudospatial
  s1 = cumul_weight.shape
  ix = (np.cumsum(np.ones(cumul_weight.shape),axis=0))*(cumul_weight <= qpoint[None,:])-1 #dimensions = stat x pseudospatial
  ix = ix.astype(int)
  q_idx = ix.max(axis=0)+1 #dimensions = pseudospatial
  q_idx[q_idx>s1[0]-1] = s1[0] - 1
  #return sorted_data,sorted_weights,cumul_weight,qpoint,ix,q_idx

  n,m = np.mgrid[0:s1[0],0:s1[1]]

  qvalue = sorted_data[q_idx[None,:],m[0,None]][0] #general case

  q_idx0 = q_idx-1 #back to ix.max
  q_idx0[q_idx<0] = 0
  qvalue[cumul_weight[q_idx0[None,:],m[0,None]][0] == qpoint] = (0.5*(sorted_data[q_idx0[None,:],m[0,None]][0] + sorted_data[q_idx[None,:],m[0,None]][0]))[cumul_weight[q_idx0[None,:],m[0,None]][0] == qpoint]

  if type(axis) == int: #restoring original shape
    qvalue = np.reshape(qvalue, (1,*s[1:]))
    qvalue = np.swapaxes(qvalue,axis,0)
    return np.squeeze(qvalue)
  else:
    return qvalue[0]




def weighted_percentile_dev(data, q, weights=None, axis=None): #adapted from weighted_stats by Jack Peterson, MIT
  """Calculate the weighted median of an array/list using numpy."""
  if weights is None:
    return np.percentile(data,q,axis=axis)
  if q == 0:
    return np.min(data,axis=axis)
  if q == 100:
    return np.max(data,axis=axis)

  if q<0 or q>100:
    raise ValueError("The percentile needs to be between 0 and 100")
  if q<1:
    print("WARNING: the percentile needs to be between 0 and 100")
  if data.shape != weights.shape:
    raise ValueError("Data and weights must have the same dimensions.")
  if np.any(np.sum(weights, axis = axis) <= 0):
    raise ValueError('Weights should have a non-zero sum.')
  if np.any(weights<0):
    raise ValueError('Negative weights are not allowed.')

  if type(axis) == int: #converting to a computation over axis 0 in 2 dimensions
    if axis != 0:
      weights1 = np.swapaxes(weights,axis,0)
      data1 = np.swapaxes(data,axis,0)
    else:
      weights1 = weights.copy()
      data1 = data.copy()
    s = weights1.shape
  else:
    weights1 = weights.flatten()
    data1 = data.flatten()
    s = weights1.shape
  #if axis != -1 and axis != weights.ndim -1:  #avoid an unnecessary swap of axes is possible
  weights1 = np.reshape(weights1,(s[0],int(np.prod(s)/s[0]))) #dimensions = stat x pseudospatial
  data1 = np.reshape(data1,(s[0],int(np.prod(s)/s[0]))) #dimensions = stat x pseudospatial

  #print(s)

  sorted_data, sorted_weights = parallel_sort(data1,weights1,axis=0)

  qpoint = q/100. * np.sum(sorted_weights, axis = 0) # q > 0 #dimensions = pseudospatial

  cumul_weight = np.cumsum(sorted_weights, axis=0) #dimensions = stat x pseudospatial
  s1 = cumul_weight.shape
  ix = np.cumsum(np.ones(cumul_weight.shape),axis=0)-1

  q_ix1 = ix.copy()
  q_ix1[cumul_weight <= qpoint[None,:]] = np.inf
  q_ix1[sorted_weights == 0] = np.inf
  q_ix1 = q_ix1.min(axis=0).astype(int) #dimensions = pseudospatial
  #ix = (cix)*(cumul_weight <= qpoint[None,:])-1 #dimensions = stat x pseudospatial
  #q_ix1 = ix[np.logical_and(cumul_weight <= qpoint[None,:], sorted_weights != 0)].max(axis=0)

  q_ix0 = ix.copy()
  q_ix0[cumul_weight > qpoint[None,:]] = -1
  q_ix0[sorted_weights == 0] = -1
  q_ix0 = q_ix0.max(axis=0).astype(int) #dimensions = pseudospatial

  #return sorted_data,sorted_weights,cumul_weight,qpoint,ix,q_ix1,q_ix0

  #q_idx0 = ix.max(axis=0) #dimensions = pseudospatial
  #q_idx1 = 0
  #q_idx[q_idx>s1[0]-1] = s1[0] - 1
  #return sorted_data,sorted_weights,cumul_weight,qpoint,ix,q_idx

  n,m = np.mgrid[0:s1[0],0:s1[1]]

  qvalue = sorted_data[q_ix1[None,:],m[0,None]][0] #general case

  #q_idx0 = q_idx-1 #back to ix.max
  #q_idx0[q_idx<0] = 0
  qvalue[cumul_weight[q_ix0[None,:],m[0,None]][0] == qpoint] = (0.5*(sorted_data[q_ix0[None,:],m[0,None]][0] + sorted_data[q_ix1[None,:],m[0,None]][0]))[cumul_weight[q_ix0[None,:],m[0,None]][0] == qpoint]

  if type(axis) == int: #restoring original shape
    qvalue = np.reshape(qvalue, (1,*s[1:]))
    qvalue = np.swapaxes(qvalue,axis,0)
    return np.squeeze(qvalue)
  else:
    return qvalue[0]
  #return sorted_data,sorted_weights,cumul_weight,qpoint,ix,q_idx,qvalue





def weighted_percentile_var(data, q, weights=None, axis=None): #adapted from weighted_stats by Jack Peterson, MIT
  """Calculate the weighted median of an array/list using numpy."""
  if weights is None:
    return np.percentile(data,q,axis=axis)
  if q == 0:
    return np.min(data,axis=axis)
  if q == 100:
    return np.max(data,axis=axis)

  if q<0 or q>100:
    raise ValueError("The percentile needs to be between 0 and 100")
  if q<1:
    print("WARNING: the percentile needs to be between 0 and 100")
  if data.shape != weights.shape:
    raise ValueError("Data and weights must have the same dimensions.")
  if np.any(np.sum(weights, axis = axis) <= 0):
    raise ValueError('Weights should have a non-zero sum.')
  if np.any(weights<0):
    raise ValueError('Negative weights are not allowed.')

  if type(axis) == int: #converting to a computation over axis 0 in 2 dimensions
    if axis != 0:
      weights1 = np.swapaxes(weights,axis,0)
      data1 = np.swapaxes(data,axis,0)
    else:
      weights1 = weights.copy()
      data1 = data.copy()
    s = weights1.shape
  else:
    weights1 = weights.flatten()
    data1 = data.flatten()
    s = weights1.shape
  weights1 = np.reshape(weights1,(s[0],int(np.prod(s)/s[0]))) #dimensions = stat x pseudospatial
  data1 = np.reshape(data1,(s[0],int(np.prod(s)/s[0]))) #dimensions = stat x pseudospatial

  sorted_data, sorted_weights = parallel_sort(data1,weights1,axis=0)
  qpoint = q/100. * np.sum(sorted_weights, axis = 0) #dimensions = pseudospatial

  cumul_weight = np.cumsum(sorted_weights, axis=0) #dimensions = stat x pseudospatial
  #perc = (cumul_weight-sorted_weights)/(cumul_weight[-1,...]-sorted_weights)

  #return sorted_data, sorted_weights, cumul_weight

  s1 = cumul_weight.shape
  ix = np.cumsum(np.ones(cumul_weight.shape),axis=0)-1

  q_ix1 = ix.copy()
  q_ix1[cumul_weight < qpoint[None,:]] = np.inf
  q_ix1[sorted_weights == 0] = np.inf
  q_ix1 = q_ix1.min(axis=0).astype(int) #dimensions = pseudospatial

  q_ix0 = ix.copy()
  q_ix0[cumul_weight > qpoint[None,:]] = -1
  q_ix0[sorted_weights == 0] = -1
  q_ix0 = q_ix0.max(axis=0).astype(int) #dimensions = pseudospatial
  q_ix0[q_ix0 == -1] = q_ix1[q_ix0 == -1]

  #interpolation: lower = ix0, upper = ix1, midpoint = (ix0+ix1)/2, nearest = ..., linear = ...
  #definition: above = cmw/cmw[-1]? , below = ???, fullspan = (cmw-cmw[0])/(cmw[-1]-cmw[0])?

  return sorted_data, sorted_weights, cumul_weight, q_ix0, q_ix1

  n,m = np.mgrid[0:s1[0],0:s1[1]]

  qvalue = sorted_data[q_ix1[None,:],m[0,None]][0] #general case

  qvalue[cumul_weight[q_ix0[None,:],m[0,None]][0] == qpoint] = (0.5*(sorted_data[q_ix0[None,:],m[0,None]][0] + sorted_data[q_ix1[None,:],m[0,None]][0]))[cumul_weight[q_ix0[None,:],m[0,None]][0] == qpoint]

  if type(axis) == int: #restoring original shape
    qvalue = np.reshape(qvalue, (1,*s[1:]))
    qvalue = np.swapaxes(qvalue,axis,0)
    return np.squeeze(qvalue)
  else:
    return qvalue[0]


#def weighted_percentile_dev(data, q, weights=None, axis=None): #adapted from weighted_stats by Jack Peterson, MIT
  #"""Calculate the weighted median of an array/list using numpy."""
  #if weights is None:
    #return np.percentile(data,q,axis=axis)

  #if q == 0:
    #return np.min(data,axis=axis)
  #if q == 100:
    #return np.max(data,axis=axis)
  #if q<0 or q>100:
    #raise ValueError("The percentile needs to be between 0 and 100")
  #if q<1:
    #print("WARNING: the percentile needs to be between 0 and 100")

  #if data.shape != weights.shape:
    #raise ValueError("Data and weights must have the same dimensions.")
  #if np.any(np.sum(weights, axis = axis) <= 0):
    #raise ValueError('Weights should have a non-zero sum.')


  #if type(axis) == int: #converting to a computation over axis 0 in 2 dimensions
    #if axis != 0:
      #weights1 = np.swapaxes(weights,axis,0)
      #data1 = np.swapaxes(data,axis,0)
    #else:
      #weights1 = weights.copy()
      #data1 = data.copy()
    #s = weights1.shape
    #weights1 = np.reshape(weights1,(s[0],int(np.prod(s)/s[0]))) #dimensions = stat x pseudospatial
    #data1 = np.reshape(data1,(s[0],int(np.prod(s)/s[0]))) #dimensions = stat x pseudospatial
  #else:
    #weights1 = weights.flatten()
    #data1 = data.flatten()
    #s = weights1.shape

  #print(s)

  #sorted_data, sorted_weights = parallel_sort(data1,weights1,axis=0)

  #qpoint = q/100. * np.sum(sorted_weights, axis = 0) # q > 0 #dimensions = pseudospatial
  #if len(s) == 1:
    #qpoint = np.array([qpoint])

  #cumul_weight = np.cumsum(sorted_weights, axis=0) #dimensions = stat x pseudospatial
  #s1 = cumul_weight.shape
  #ix = (np.cumsum(np.ones(cumul_weight.shape),axis=0))*(cumul_weight <= qpoint[None,:])-1 #dimensions = stat x pseudospatial
  #ix = ix.astype(int)
  #q_idx = ix.max(axis=0)+1 #dimensions = pseudospatial
  #q_idx[q_idx>s1[0]-1] = s1[0] - 1
  #return sorted_data,sorted_weights,cumul_weight,qpoint,ix,q_idx
  #n,m = np.mgrid[0:s1[0],0:s1[1]]

  #qvalue = sorted_data[q_idx[None,:],m[0,None]][0] #general case

  #q_idx0 = q_idx-1 #back to ix.max
  #q_idx0[q_idx<0] = 0
  #qvalue[cumul_weight[q_idx0[None,:],m[0,None]][0] == qpoint] = (0.5*(sorted_data[q_idx0[None,:],m[0,None]][0] + sorted_data[q_idx[None,:],m[0,None]][0]))[cumul_weight[q_idx0[None,:],m[0,None]][0] == qpoint]
  #return sorted_data,sorted_weights,cumul_weight,qpoint,ix,q_idx,qvalue



  #if cumul_weight.shape[0] == 1:
    #cumul_weight = cumul_weight[0,:]
    #qpoint = qpoint[0]
    #ix = np.arange(1,len(cumul_weight)+1)*(cumul_weight <= qpoint)-1
    #ix = ix.astype(int)
    #below_q_idx = ix.max()
    #qvalue = 0
    #if cumul_weight[below_q_idx] == qpoint:
      #qvalue = sorted_data[0][below_q_idx]
    #elif cumul_weight[below_q_idx] < qpoint:
      #qvalue = sorted_data[0][below_q_idx] + (sorted_data[0][below_q_idx+1]-sorted_data[0][below_q_idx])*(qpoint-cumul_weight[below_q_idx])/(cumul_weight[below_q_idx+1]-cumul_weight[below_q_idx])
    #elif cumul_weight[below_q_idx] == 0:
      #qvalue = sorted_data[0][below_q_idx+1]
    #return qvalue
  #else: #transpose for easier processing
    #ix = (np.cumsum(np.ones(cumul_weight.shape),axis=-1))*(cumul_weight <= qpoint[:,None])-1
    #ix = (ix.astype(int)).transpose() #dimensions = stat x pseudospatial
    #print(ix)
    #cumul_weight = cumul_weight.transpose() #dimensions = stat x pseudospatial
    #sorted_data = sorted_data.transpose() #dimensions = stat x pseudospatial
    #below_q_idx = ix.max(axis=0) #dimensions = pseudospatial
    #qvalue = np.zeros(ix.shape[1]) #dimensions = pseudospatial
    #print(cumul_weight)
    #print(below_q_idx)
    #print(qvalue)
    #print(qpoint)
    #input()
    #n,m = np.mgrid[0:cumul_weight.shape[0],0:cumul_weight.shape[1]]
    #print(cumul_weight[below_q_idx[None,:],m[0,None]][0])
    #print(cumul_weight[below_q_idx[None,:],m[0,None]][0] == qpoint)
    #print(cumul_weight[below_q_idx[None,:],m[0,None]][0] < qpoint)
    #input('')
    #print(sorted_data[below_q_idx[None,:],m[0,None]][0])
    #input('')

    #qvalue[cumul_weight[below_q_idx[None,:],m[0,None]][0] == qpoint] = (sorted_data[below_q_idx[None,:],m[0,None]][0])[cumul_weight[below_q_idx[None,:],m[0,None]][0] == qpoint]
    #qvalue[cumul_weight[below_q_idx[None,:],m[0,None]][0] < qpoint] = (sorted_data[below_q_idx[None,:],m[0,None]][0] + (sorted_data[below_q_idx[None,:]+1,m[0,None]][0]-sorted_data[below_q_idx[None,:],m[0,None]][0])*(qpoint-cumul_weight[below_q_idx[None,:],m[0,None]][0])/(cumul_weight[below_q_idx[None,:]+1,m[0,None]][0]-cumul_weight[below_q_idx[None,:],m[0,None]][0]))[cumul_weight[below_q_idx[None,:],m[0,None]][0] < qpoint]
    #qvalue[cumul_weight[below_q_idx[None,:],m[0,None]][0] == 0] = (sorted_data[below_q_idx[None,:]+1,m[0,None]][0])[cumul_weight[below_q_idx[None,:],m[0,None]][0] == 0]

    #qvalue = np.reshape(qvalue.transpose(),s[:-1])
    #return qvalue



#def weighted_percentile(data, q, weights=None, axis=None): #adapted from weighted_stats by Jack Peterson, MIT
  #"""Calculate the weighted median of an array/list using numpy."""
  #if weights is None:
    #return np.percentile(data,q,axis=axis)

  #if q == 0:
     #return np.min(data,axis=axis)
  #if q<0 or q>100:
    #raise ValueError("The percentile needs to be between 0 and 100")
  #if q<1:
    #print("WARNING: the percentile needs to be between 0 and 100")

  #if data.shape != weights.shape:
    #raise ValueError("Data and weights must have the same dimensions.")
  #if np.any(np.sum(weights, axis = axis) <= 0):
    #raise ValueError('Weights should have a non-zero sum.')
    ##if type(axis) == int: #converting a sort along a generic axis in N dimensions to a sort along the last axis in 2 dimensions
      ##data1 = np.swapaxes(data,axis,-1)
      ##weights1 = np.swapaxes(weights1,axis,-1)
      ##s = data1.shape
      ##data1 = np.reshape(data1,(np.prod(s[-1]),s[-1]))
      ##weights1 = np.reshape(weights1,(np.prod(s[-1]),s[-1]))
    ##else:
      ##data1 = data.flatten()
      ##weights1 = weights.flatten()
    ##sorted_ind = argsort(weights1, axis=-1)
    ##n = mgrid[0:data1.shape[0],0:data1.shape[1]][0]
    ##sorted_data = data1[n,sorted_ind]
    ##sorted_weights = weights1[n,sorted_ind]
    ##if type(axis) == int: #converting a sort along a generic axis in N dimensions to a sort along the last axis in 2 dimensions
      ##sorted_data = np.reshape(sorted_data,s)
      ##sorted_weights = np.reshape(sorted_weights,s)
      ##sorted_data = np.swapaxes(sorted_data,axis,-1)
      ##sorted_weights = np.swapaxes(sorted_weights,axis,-1)

  #if type(axis) == int: #converting a case of generic axis in N dimensions to a case of last axis in 2 dimensions
    #if axis != -1 and axis != weights.ndim -1:  #avoid an unnecessary swap of axes is possible
      #weights1 = np.swapaxes(weights,axis,-1)
      #data1 = np.swapaxes(data,axis,-1)
    #else:
      #weights1 = weights.copy()
      #data1 = data.copy()
    #s = weights1.shape
    #if len(s)>2:  #avoid an unnecessary reshaping
      #weights1 = np.reshape(weights1,(int(np.prod(s[:-1])),s[-1])) #dimensions = pseudospatial x stat
      #data1 = np.reshape(data1,(int(np.prod(s[:-1])),s[-1])) #dimensions = pseudospatial x stat
  #else:
    #weights1 = weights.flatten()
    #data1 = data.flatten()

  #sorted_data, sorted_weights = parallel_sort(data1,weights1,axis=-1)

  #print(data1)
  #print(weights1)

  #sorted_data, sorted_weights = parallel_sort(data1,weights1,axis=0)

  #print(sorted_data)
  #print(sorted_weights)
  #return sorted_data,sorted_weights

  ##qpoint = np.transpose([q/100. * np.sum(sorted_weights, axis = -1)]) # q > 0 #dimensions = pseudospatial x 1
  #qpoint = q/100. * np.sum(sorted_weights, axis = -1) # q > 0 #dimensions = pseudospatial
  #cumul_weight = np.cumsum(sorted_weights, axis=-1) #dimensions = pseudospatial x stat
  #if cumul_weight.shape[0] == 1:
    #cumul_weight = cumul_weight[0,:]
    #qpoint = qpoint[0]
    #ix = np.arange(1,len(cumul_weight)+1)*(cumul_weight <= qpoint)-1
    #ix = ix.astype(int)
    #below_q_idx = ix.max()
    #qvalue = 0
    #if cumul_weight[below_q_idx] == qpoint:
      #qvalue = sorted_data[0][below_q_idx]
    #elif cumul_weight[below_q_idx] < qpoint:
      #qvalue = sorted_data[0][below_q_idx] + (sorted_data[0][below_q_idx+1]-sorted_data[0][below_q_idx])*(qpoint-cumul_weight[below_q_idx])/(cumul_weight[below_q_idx+1]-cumul_weight[below_q_idx])
    #elif cumul_weight[below_q_idx] == 0:
      #qvalue = sorted_data[0][below_q_idx+1]
    #return qvalue
  #else: #transpose for easier processing
    #ix = (np.cumsum(np.ones(cumul_weight.shape),axis=-1))*(cumul_weight <= qpoint[:,None])-1
    #ix = (ix.astype(int)).transpose() #dimensions = stat x pseudospatial
    #print(ix)
    #cumul_weight = cumul_weight.transpose() #dimensions = stat x pseudospatial
    #sorted_data = sorted_data.transpose() #dimensions = stat x pseudospatial
    #below_q_idx = ix.max(axis=0) #dimensions = pseudospatial
    #qvalue = np.zeros(ix.shape[1]) #dimensions = pseudospatial
    #print(cumul_weight)
    #print(below_q_idx)
    #print(qvalue)
    #print(qpoint)
    #input()
    #n,m = np.mgrid[0:cumul_weight.shape[0],0:cumul_weight.shape[1]]
    #print(cumul_weight[below_q_idx[None,:],m[0,None]][0])
    #print(cumul_weight[below_q_idx[None,:],m[0,None]][0] == qpoint)
    #print(cumul_weight[below_q_idx[None,:],m[0,None]][0] < qpoint)
    #input('')
    #print(sorted_data[below_q_idx[None,:],m[0,None]][0])
    #input('')

    #qvalue[cumul_weight[below_q_idx[None,:],m[0,None]][0] == qpoint] = (sorted_data[below_q_idx[None,:],m[0,None]][0])[cumul_weight[below_q_idx[None,:],m[0,None]][0] == qpoint]
    #qvalue[cumul_weight[below_q_idx[None,:],m[0,None]][0] < qpoint] = (sorted_data[below_q_idx[None,:],m[0,None]][0] + (sorted_data[below_q_idx[None,:]+1,m[0,None]][0]-sorted_data[below_q_idx[None,:],m[0,None]][0])*(qpoint-cumul_weight[below_q_idx[None,:],m[0,None]][0])/(cumul_weight[below_q_idx[None,:]+1,m[0,None]][0]-cumul_weight[below_q_idx[None,:],m[0,None]][0]))[cumul_weight[below_q_idx[None,:],m[0,None]][0] < qpoint]
    #qvalue[cumul_weight[below_q_idx[None,:],m[0,None]][0] == 0] = (sorted_data[below_q_idx[None,:]+1,m[0,None]][0])[cumul_weight[below_q_idx[None,:],m[0,None]][0] == 0]

    #qvalue = np.reshape(qvalue.transpose(),s[:-1])
    #return qvalue



#def weighted_percentile_1(data, q, weights=None, axis=None): #adapted from weighted_stats by Jack Peterson, MIT
  #"""Calculate the weighted median of an array/list using numpy."""
  #if weights is None:
    #return np.percentile(data,q,axis=axis)

  #if q == 0:
     #return np.min(data,axis=axis)
  #if q<0 or q>100:
    #raise ValueError("The percentile needs to be between 0 and 100")
  #if q<1:
    #print("WARNING: the percentile needs to be between 0 and 100")

  #if data.shape != weights.shape:
    #raise ValueError("Data and weights must have the same dimensions.")
  #if np.any(np.sum(weights, axis = axis) <= 0):
    #raise ValueError('Weights should have a non-zero sum.')

  #if type(axis) == int: #converting a sort along a generic axis in N dimensions to a sort along the last axis in 2 dimensions
    #if axis != -1 and axis != weights.ndim -1:
      #weights1 = np.swapaxes(weights,axis,-1)
      #data1 = np.swapaxes(data,axis,-1)
    #else:
      #weights1 = weights.copy()
      #data1 = data.copy()
  #else:
    #weights1 = weights.flatten()
    #data1 = data.flatten()
  #s = weights1.shape
  #weights1 = np.reshape(weights1,(int(np.prod(s[:-1])),s[-1])) #dimensions = pseudospatial x stat
  #data1 = np.reshape(data1,(int(np.prod(s[:-1])),s[-1])) #dimensions = pseudospatial x stat
  #sorted_data, sorted_weights = parallel_sort(data1,weights1,axis=-1)

  ##inverting the axes to deal more easily with the 1D case
  #qpoint = q/100. * np.sum(sorted_weights, axis = -1) # q > 0 #dimensions = pseudospatial
  #cumul_weight = np.transpose(np.cumsum(sorted_weights, axis=-1)) #dimensions = stat x pseudospatial
  #if cumul_weight.shape[1] == 1:
    #cumul_weight = cumul_weight[:,0]
  #ix = (np.cumsum(np.ones(cumul_weight.shape),axis=0))*(cumul_weight <= qpoint)-1
  #ix = ix.astype(int) #dimensions = stat x pseudospatial
  #below_q_idx = ix.max(axis=0) #dimensions = pseudospatial

  #qvalue = np.zeros(below_q_idx.shape)
  ##n = np.mgrid[0:data1.shape[0],0:data1.shape[1]][0]
  ##print("(cumul_weight[n[:,0:1],below_q_idx] == qpoint)", (cumul_weight[n[:,0:1],below_q_idx] == qpoint))
  ##input('')
  ##print(sorted_data[n[:,0:1],below_q_idx])
  ##input('')
  #print(qpoint,cumul_weight,below_q_idx,qvalue)

  #n = np.mgrid[0:cumul_weight.shape[0],0:cumul_weight.shape[1]][0]
  #print(n)
  #print("(cumul_weight[n[:,0:1],below_q_idx] == qpoint)", (cumul_weight[n[:,0:1],below_q_idx] == qpoint))
  #input('')

  #qvalue[a == qpoint] = sorted_data[n[:,0:1],below_q_idx]
  #qvalue[cumul_weight[n[:,0:1],below_q_idx] < qpoint] = sorted_data[n[:,0:1],below_q_idx] + (sorted_data[n[:,0:1],below_q_idx+1]-sorted_data[n[:,0:1],below_q_idx])*(qpoint-cumul_weight[n[:,0:1],below_q_idx])/(cumul_weight[n[:,0:1],below_q_idx+1]-cumul_weight[n[:,0:1],below_q_idx])
  #qvalue[cumul_weight[n[:,0:1],below_q_idx] == 0] = sorted_data[n[:,0:1],below_q_idx+1]

  #qvalue = np.reshape(weights1,(np.prod(s[:-1]),s[-1]))

  #return qvalue


def parallel_sort(a_ref,a_passive,axis=-1):
  """Reorders a passive array following the sorting order of a reference array.

  Parameters
  -----
  a_ref: ndarray
    The array containing the data along which to sort.
  a_passive: ndarray
    Of the same size as a_ref, this data will be passively sorted according to
    the sorting order of a_ref.

  Options
  -----
  axis: int
    Axis along which the sorting is performed (default = -1). If None, the sort
    is performed on flattened arrays

  Returns
  -----
  sorted_ref: ndarray (or 1darray)
    The data from a_ref, sorted along the specified axis (or flattened and
    sorted)
  sorted_passive: ndarray (or 1darray)
    The data from a_passive, sorted following the order of the specified axis
    of a_ref (or flattened and sorted following the order of the flattened
    a_ref)
  """
  if a_ref.shape != a_passive.shape:
    raise ValueError("Both arrays must have the same dimensions.")
  if a_ref.ndim == 1:
    sorted_ind = np.argsort(a_ref)
    return a_ref[sorted_ind],a_passive[sorted_ind]
  if type(axis) == int: #converting a sort along a generic axis in N dimensions to a sort along the last axis in 2 dimensions
    if axis != -1 and axis != a_ref.ndim -1:
      a_ref1 = np.swapaxes(a_ref,axis,-1)
      a_passive1 = np.swapaxes(a_passive,axis,-1)
    else:
      a_ref1 = a_ref.copy()
      a_passive1 = a_passive.copy()
    s = a_ref1.shape
    if len(s) > 2:
      a_ref1 = np.reshape(a_ref1,(np.prod(s[:-1]),s[-1]))
      a_passive1 = np.reshape(a_passive1,(np.prod(s[:-1]),s[-1]))
  else:
    a_ref1 = a_ref.flatten()
    a_passive1 = a_passive.flatten()
  sorted_ind = np.argsort(a_ref1, axis=-1)
  if type(axis) == int:
    n = np.mgrid[0:a_ref1.shape[0],0:a_ref1.shape[1]][0]
    sorted_ref = a_ref1[n,sorted_ind]
    sorted_passive = a_passive1[n,sorted_ind]
  else:
    sorted_ref = a_ref1[sorted_ind]
    sorted_passive = a_passive1[sorted_ind]
  if type(axis) == int: #back to the initial shape and axis order
    if len(s) > 2:
      sorted_ref = np.reshape(sorted_ref,s)
      sorted_passive = np.reshape(sorted_passive,s)
    if axis != -1 and axis != a_ref.ndim -1:
      sorted_ref = np.swapaxes(sorted_ref,axis,-1)
      sorted_passive = np.swapaxes(sorted_passive,axis,-1)
  return sorted_ref, sorted_passive

# find and extract objects in labelled coldens contour map: scipy.ndimage.measurements.find_objects
# for each object:
#   box-counting fractal dimension for each object found => fractal dimension of the 2D projection fdim2d
#   infer the fractal dimension in 3D fdim3d (from tabulated or analytic expression)
#   infer total volume => volume density increment
#       (other fancier option: generate a random fractal of dimension fdim3d with projection matching the contour, apply its voldens to the cube)
# AstroDendro: significance of the peaks?
# measuring the elongation of subregions : skimage.morphology : medial axis, skeletonization, thinning
# Smooth the contours to avoid jagged skeleton ?
# Link between elongation and skeleton length ? major/minor vs area vs skimage sum ?
