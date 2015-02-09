'''
Utilities to read audio files
'''

# These are only needed for wavreading and downsampling
import scipy.io.wavfile as wavfile
import scipy.signal
#
import numpy as np
import os

def wavread(file_name):
    '''
    Wrapper for scipy's wavread function.

    Watch out: It inverts argument order and casts signal to float!
    '''
    fs, x_t = wavfile.read(file_name)
    return (x_t.astype(float), fs)

def rawread(file_name, byteorder_raw='littleendian'):
    '''
    Read audio signal in raw format
    '''
    if byteorder_raw == 'littleendian':
        dt = '<h'
    elif byteorder_raw == 'bigendian':     
        dt = '>h'
    else:
        raise ValueError, "Unknown byteorder_raw %s" % byteorder_raw  
    # This does no serious check for bad formatted inputs!!
    x = np.fromfile(open(file_name, 'rb'), dt)
    x = np.cast['float64'](x)
    
    return x
    
def rawwrite(file_name, x):
    '''
    Write audio signal in raw format
    '''
    # Adjust bundaries to max in int16
    max_array_value = np.max(np.absolute(x))
    max_int16       = np.iinfo(np.int16).max
    x               = np.ceil(x * (0.99 * max_int16/max_array_value))
    x               = np.cast['int16'](x)
    x.tofile(open(file_name, 'wb'), format='%h')
    
    return 1

def read(file_name, in_fs=None, out_fs=None, byteorder_raw='littleendian'):
    '''
    Tries to pick appropiate method based on file type, ensures fs correct
    and donwsamples to appropiate working frequency if solicited.
    '''

    # Check files exist
    if not os.path.exists(file_name):
        raise IOError, "Can not open file_name %s" % file_name
    
    # Read audio in the desired format
    audio_type = os.path.basename(file_name).split('.')[-1]
    if audio_type == 'raw' or audio_type == 'pcm':
        # Enforce providing frequency
        if not in_fs:
            raise ValueError, ("For raw/pcm files you need to specify the"
                              " input frequency")
        y = rawread(file_name, byteorder_raw=byteorder_raw)

    elif audio_type == 'wav':
        [in_fs_wav, y] = wavfile.read(file_name)
        # Ensure float
        y              = y.astype(float)
        # If frequency indicated, check it matches
        if in_fs and in_fs_wav != in_fs:
            raise ValueError, ("You specified a sampling freq. with -fs %d, "
                               " but the file has fs %d") % (in_fs, in_fs_wav)
        in_fs = in_fs_wav

    else:
        raise IOError, "Unknown file type %s" % audio_type

    # Resample to work_fs if solicited
    if not out_fs:
        out_fs = in_fs    
    else:
        if out_fs < in_fs:
            y = scipy.signal.decimate(y, in_fs/out_fs)
        elif out_fs > in_fs:
            raise ValueError, ("Your work sampling frequency is in fact larger"
                              " than the fs of the file!")
    
    return [y, out_fs]
