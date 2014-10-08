'''
Interfaces to the DIRHA project data
'''

import re
import os
import numpy as np

# Ref mics for each room
REF_MICS = { 'BEDROOM':  'Wall/B1L.wav', 'LIVINGROOM': 'Array/LA6.wav', 
             'KITCHEN': 'Array/KA3.wav', 'BATHROOM': 'Wall/R1R.wav', 
             'CORRIDOR' : 'Wall/C1R.wav' } 

# DIRHA corpus matching regexp
DIRHA_SIM  = re.compile('.*DIRHA_sim2/([A-z]+)/([a-z0-9]+)/sim([0-9]+)/'
                        'Signals/Mixed_Sources/([A-Za-z]+)/([A-Za-z]+)/'
                        '([A-Z0-9]+)\.([^\.]*)$') 
DIRHA_GRID = re.compile('.*grid_dirha/([a-z0-9]+)/sim([0-9]+)/Signals/'
                        'Mixed_Sources/([A-Za-z]+)/([A-Za-z]+)/([A-Z0-9]+)'
                        '\.([^\.]*)$') 
EXTRACT_RE = re.compile('(.*)([A-z]+)/([a-z0-9]+)/sim([0-9]+)/Signals/'
                        'Mixed_Sources/([A-Za-z]+)/([A-Za-z]+)/([A-Z0-9]+)'
                        '\.([^\.]*)$')


def readmetadata(txt_path, in_fs=None, work_fs=None):
    '''
    Reads a DIRHA simulated corpus meta-data file and returns the global and
    source tags as dictionaries

    Input: txt_path  string pointing to the DIRHA simulated corpus *.txt
                     meta-data file for a given *.wav microphone file.
                     The *.wav file can also be given instead. 

    Input: in_fs     int indicating sampling frequency we of the DIRHA corpus

    Input: work_fs   int indicating sampling frequency we will work in.
                     Not that DIRHA_SIMII uses 48KHz wereas we work usually
                     at 16KHz  
    '''

    # DEDUCE SAMPLING FREQS IF NOT GIVEN
    if not in_fs:
        if DIRHA_SIM.match(txt_path): 
            in_fs = 48000
        elif DIRHA_GRID.match(txt_path): 
            in_fs = 16000 
        else: 
            raise EnvironmentError, ("%s does not seem to be a DIRHA corpus"
                                     " file" % txt_path)
    if not work_fs:
        work_fs = in_fs 

    # READ METADATA FILE
    # Read file content
    with open(txt_path) as f:
        lines_txt = f.readlines()
    # Initialization 
    events   = []
    sources  = []
    state    = ''
    i        = 0
    n_lines  = len(lines_txt)
    # State machine reading all fields of the file into global and 
    # source lists  
    while i < n_lines:
        line = lines_txt[i].rstrip() 
        # On idle state, enter other states when pertinent
        if not state:
            if line == '<GLOBAL>':
                state   =  'global'
            elif line == '<SOURCE>':
                state   = 'source' 
                # Start a dictionary for this source
                source  = {}
                # Start by storing the current txt_file
                source['file'] = txt_path 
            else:
                pass
        # On global state
        elif state == 'global':
            # Exit
            if line == '</GLOBAL>':
                state = ''
            else:
                events.append(line.rstrip().split()) 
        # On local state
        elif state == 'source':
            # Exit
            if line == '</SOURCE>':
                # Append dictionary
                sources.append(source)
                state = ''
            # Tag found 
            else:
                mt = re.search('\s*<([^>]*)>([^>]*)<.*', line)
                # tag value /tag format 
                if mt:
                    # Sample specifications (may need to be downsampled)
                    if mt.group(1) == 'begin_sample':
                        if in_fs > work_fs:
                            source[mt.group(1)] = dsmp(int(mt.group(2)),
                                                       in_fs,
                                                       work_fs)
                        else: 
                            source[mt.group(1)] = int(mt.group(2))
                    elif mt.group(1) == 'end_sample':
                        if in_fs > work_fs:
                            source[mt.group(1)] = dsmp(int(mt.group(2)),
                                                       in_fs,
                                                       work_fs)
                        else: 
                            source[mt.group(1)] = int(mt.group(2))
                    # other
                    else: 
                        source[mt.group(1)] = mt.group(2)
                # tag or /tag format
                elif re.match('\s*<label=\w*>', line):
                    # Read 
                    label = re.search('<label=(.*)>', line).group(1)
                    source[label] = []
                    while not re.match('\s*</label=\w*>', line):
                        i += 1
                        line = lines_txt[i].rstrip().lstrip()
                        if not re.match('\s*</label=\w*>', line):
                            # Downsample the sample references
                            if in_fs > work_fs:
                                pieces = line.split() 
                                source[label].append(
                                    [dsmp(int(pieces[0]), in_fs, work_fs), 
                                     dsmp(int(pieces[1]), in_fs, work_fs)] 
                                    + pieces[2:])
                            else:
                                pieces = line.split() 
                                source[label].append([int(pieces[0]),
                                                      int(pieces[1])]
                                                     + pieces[2:])
                else:
                    raise ValueError, ("Could not parse SOURCE line:\n\n%s"
                                       % line) 
        # Read 
        i += 1

    # SANITY CHECK: WE READ SOMETHING
    if not len(sources):
        raise EnvironmentError, ("File %s does not seem to be a DIRHA"
                                 " meta-data file" % (txt_path))

    return (events, sources)


def dsmp(T, in_fs, work_fs):
    '''
    Adjusts time measured in samples to match the working sampling 
    frequency 
    '''
    return int(np.floor(T*work_fs*1.0/in_fs))


class DirhaMicMetaData():
    '''
    Stores the metadata for one microphone of a DIRHA simulated corpus given
    its path. 
    '''

    def __init__(self, txt_path, work_fs):
        '''
        Input: txt_path  string pointing to the DIRHA simulated corpus *.txt
                         meta-data file for a given *.wav microphone file.
                         The *.wav file can also be given instead. 

        Input: work_fs   int indicating sampling frequency we will work in.
                         Not that DIRHA_SIMII uses 48KHz wereas we work usually
                         at 16KHz  
        '''

        # Admit also the wav path instead of txt
        if re.match('.*\.wav$', txt_path):
            txt_path = re.sub('\.wav$', '.txt', txt_path)

        self.txt_path = txt_path

        # CHECK FOR PATH EXISTING
        if not os.path.isfile(txt_path):
            raise IOError, "Could not find ground truth file %s" % txt_path


        # CHECK FOR DIRHA CORPUS
        # Set also input frequency accordingly
        if DIRHA_SIM.match(txt_path): 
            in_fs = 48000
        elif DIRHA_GRID.match(txt_path): 
            in_fs = 16000 
        else: 
            raise EnvironmentError, ("%s does not seem to be a DIRHA corpus"
                                     " file" % txt_path)

        # EXTRACT CORPUS CHARACTERISTICS FROM METADATA FILE NAME
        [self.root, self.lang, self.sset, sim, self.room, self.device, 
         self.mic, self.ftype] =  EXTRACT_RE.search(txt_path).groups() 
        self.sim = int(sim)
       
        # Extract data from the file
        self.glob, self.sources = readmetadata(txt_path, in_fs=in_fs, 
                                               work_fs=work_fs)


    def get_ref_mic_source(self, src):
        '''
        Returns the reference microphone in the room were this src took 
        place for this simulation.
        ''' 
        # Get room
        room = src['pos'].split('_')[-1]
        # Fix for short-name notation
        if room == 'BAT':
            room = 'BATHROOM'
        elif room == 'BEDROO' or room == 'BED':
            room = 'BEDROOM';
        elif room == 'CORR':
            room = 'CORRIDOR';
        elif room == 'KIT':
            room =  'KITCHEN';
        elif room == 'LIV':
            room = 'LIVINGROOM';
        return (self.txt_path.split('/Mixed_Sources/')[0] 
                + '/Mixed_Sources/' + room[0]  + room[1:].lower() 
                +  '/' + REF_MICS[room]) 

    def get_source(self, name):
        '''
        Return source with given name
        ''' 
        found = None
        for s in self.sources: 
            if s['name'] == name:
                return s
        if not found:
            raise EnvironmentError, ("source %s could not be located in "
                                     "meta-data file %s" % (name, 
                                                            self.txt_path))

    def get_sources_from_ref_mic(self, regexp_fiter='sp_.*'):

        '''
        Returns speech events as seen from the respective reference 
        microphones in the room where they take place.

        Note that this is independent of the input mic we read.  
        '''

        sources = {}
        # Loop over all sources, filter speech ones
        for src in self.sources:
            if re.match(regexp_fiter, src['name']):
                # Get this speech event as seen from its reference microphone
                # and append it 
                ref_mic_path         = self.get_ref_mic_source(src) 
                dirha_or_sources     = DirhaMicMetaData(ref_mic_path, 16000)
                sources[src['name']] = dirha_or_sources.get_source(src['name']) 

        return sources
