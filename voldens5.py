#import orionfits as of
import scipy.ndimage.measurements as imm
from scipy.interpolate import interp1d
import numpy as np
from os import mkdir
try:
    import astropy.io.fits as pf
except ImportError:
    import pyfits as pf


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
  raise NotImplementedError("Working on it!")
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

def save_simple(voldensities,volumes,name,header,stat="peak",w="vol"):
  """Saves a simple, compressed view of the volume density structure. The
  global PDF of volume density (densities and corresponding volumes) is saved
  as a CSV file, along with a representative 2D map of volume density values.

  Parameters
  -----
  voldensities: 3darray
    Volume densities found along each line of sight (in some unit per cm-3)
  volumes: 3darray
    Volumes corresponding to each volume density (in pc3)
  name: str
    File name. The PDF is saved to name-PDF.txt, the map to name-stat.fits,
    where 'stat' is the selected statistic.
  header: pyfits.Header object
    Header (typically of column density data) which will be altered and used
    as the header of the output FITS file.

  Options
  -----
  stat: str
    Statistic computed for each line of sight to generated the returned map.
    Can be "peak" (maximum along the line of sight), "mean" or "median"
    (default: "peak")
  w: str
    Weighting scheme for the computation of a "mean" or "median" map. Can be
    "vol" (volume-weighted) or "mass" (mass-weighted) (default: "vol")
  """
  if stat=='median':
    save_stat_map(voldensities,volumes,name,header,stat='perc',w=w,p=50)
  else:
    save_stat_map(voldensities,volumes,name,header,stat=stat,w=w)

  voldensitiesflat,volumesflat = cube_to_pdf(voldensities,volumes)
  f = open(name+'-pdf.txt',mode="w")
  f.write('Density\tVolume\n')
  f.write('cm-3\tpc3\n\n')
  for i in range(len(voldensitiesflat)):
    l = '{:.8e}\t{:.8e}\n'.format(voldensitiesflat[i],volumesflat[i])
    f.write(l)
  f.close()

def save_stat_map(voldensities,volumes,name,header,stat="peak",w="vol",p=50):
  """Computes and saves a map of a simple statistic computed indenpendently along each
  line of sight.

  Parameters
  -----
  voldensities: 3darray
    Volume densities found along each line of sight (in some unit per cm-3)
  volumes: 3darray
    Volumes corresponding to each volume density (in pc3)
  name: str
    File name. The map is saved to name-w-stat-p.fits, where 'w', 'stat', 'p'
    are the selected options where relevant.
  header: pyfits.Header object
    Header (typically of column density data) which will be altered and used
    as the header of the output FITS file.


  Options
  -----
  stat: str
    Statistic computed for each line of sight to generate the returned map.
    Can be "peak" (maximum along the line of sight), "mean", "std" (standard
    deviation), "median", or "perc" (percentile) (default: "peak")
  w: str
    Weighting scheme for the computation of a "mean" or "median" map. Can be
    "vol" (volume-weighted) or "mass" (mass-weighted) (default: "vol")
  p: float
    Percentile selected for the "perc" option, between 0 and 100 (defalut: 50,
    corresponding to the median)
  """
  if w == "vol":
    wei = volumes
  elif w == "mass":
    wei = volumes*voldensities
  else:
    raise ValueError('Weighting must be by volume (w="vol") or by mass (w="mass").')

  if stat == "peak":
    statmap = voldensities.max(-1)
    filename = name+"-peak"
  elif stat == "mean":
    statmap = np.average(voldensities,axis=-1,weights=wei)
    filename = name+"-"+w+"-mean"
  elif stat == "std":
    mm = np.average(voldensities,axis=-1,weights=wei)
    statmap = np.sqrt(np.average((voldensities-mm[:,:,None])**2,axis=-1,weights=wei))
    filename = name+"-"+w+"-std"
  elif stat == "perc":
    raise NotImplementedError("Working on it!")
    statmap = weighted_percentile(voldensities,q = p, weights=wei, axis=-1)
    if p == 50:
      filename = name+"-"+w+"-median"
    else:
      filename = name+"-"+w+"-perc-"+str(int(p))
  else:
    raise ValueError('Choose a stat among "peak", "mean", "std" or "perc".')

  head = header.copy()
  if 'DATAMIN' in head.keys():
    head['DATAMIN'] = (statmap.min(),head.comments['DATAMIN'])
  if 'DATAMAX' in head.keys():
    head['DATAMAX'] = (statmap.max(),head.comments['DATAMAX'])
  if 'BUNIT' in head.keys():
    head['BUNIT'] = ('cm-3',head.comments['BUNIT'])
  pf.writeto(filename+'.fits',statmap,head) #fix it

def stat_map(voldensities,volumes,stat="peak",w="vol",p=50):
  """Computes a map of a simple statistic computed indenpendently along each
  line of sight.

  Parameters
  -----
  voldensities: 3darray
    Volume densities found along each line of sight (in some unit per cm-3)
  volumes: 3darray
    Volumes corresponding to each volume density (in pc3)
  name: str
    File name. The map is saved to name-w-stat-p.fits, where 'w', 'stat', 'p'
    are the selected options where relevant.

  Options
  -----
  stat: str
    Statistic applied to eacn line of sight to generated the returned map. Can
    be "peak" (maximum along the line of sight), "mean", "std" (standard
    deviation) or "perc" (percentile) (default: "peak")
  w: str
    Weighting scheme for the computation of a "mean" or "median" map. Can be
    "vol" (volume-weighted) or "mass" (mass-weighted) (default: "vol")
  p: float
    Percentile selected for the "perc" option, between 0 and 100 (defalut: 50,
    corresponding to the median)
  """
  if w == "vol":
    wei = volumes
  elif w == "mass":
    wei = volumes*voldensities
  else:
    raise ValueError('Weighting must be by volume (w="vol") or by mass (w="mass").')

  if stat == "peak":
    statmap = voldensities.max(-1)
  elif stat == "mean":
    statmap = np.average(np.ma.masked_array(voldensities,mask=wei==0),axis=-1,weights=np.ma.masked_array(wei,mask=wei==0))
    statmap = statmap.filled(0)
  elif stat == "std":
    mm = np.average(np.ma.masked_array(voldensities,mask=wei==0),axis=-1,weights=np.ma.masked_array(wei,mask=wei==0))
    statmap = np.sqrt(np.average((voldensities-mm[:,:,None])**2,axis=-1,weights=np.ma.masked_array(wei,mask=wei==0)))
    statmap = statmap.filled(0)
  elif stat == "median":
    raise NotImplementedError("Working on it!")
    statmap = weighted_percentile(voldensities,q = 50, weights=wei, axis=-1)
  elif stat == "perc":
    raise NotImplementedError("Working on it!")
    statmap = weighted_percentile(voldensities,q = p, weights=wei, axis=-1)
  else:
    raise ValueError('Choose a stat among "peak", "mean", "std", "median" or "perc".')
  return np.ma.masked_array(statmap,mask=wei.sum(-1)==0)

def cube_to_pdf(voldensities,volumes,save=False,name=None):
  """Extracts the global probability distribution function (PDF) from all the
  lines of sight of a volume density (with associated volulmes) cube.

  Parameters
  -----
  voldensities: 3darray
    Volume densities found along each line of sight (in some unit per cm-3)
  volumes: 3darray
    Volumes corresponding to each volume density (in pc3)

  Options
  -----
  save: bool
    If True, saves the PDF (volume densities and volumes) to a text file
    (default: False)
  name: str or None
    Prefix to the pdf.txt created if 'save' is True (default: None)

  Returns
  -----
  voldensflat: 1darray
    Set of all volume density values found in the original cube
  volflat: 1darray
    Total volume associated with each volume density value throughout the
    entire cube
  """
  #TODO compress input if masked
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


