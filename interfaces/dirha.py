'''
Interfaces to the DIRHA project data
'''

import re
import os
import numpy as np

##############
#  CONSTANTS
##############

# DIRHA microphone list
MIC_LIST = [ 'Bathroom/Wall/R1C.wav', 'Bathroom/Wall/R1L.wav',
             'Bathroom/Wall/R1R.wav', 'Bedroom/Wall/B1L.wav',
             'Bedroom/Wall/B1R.wav',  'Bedroom/Wall/B2C.wav',
             'Bedroom/Wall/B2L.wav',  'Bedroom/Wall/B2R.wav',
             'Bedroom/Wall/B3L.wav',  'Bedroom/Wall/B3R.wav',
             'Corridor/Wall/C1L.wav', 'Corridor/Wall/C1R.wav',
             'Kitchen/Array/KA1.wav', 'Kitchen/Array/KA2.wav',
             'Kitchen/Array/KA3.wav', 'Kitchen/Array/KA4.wav',
             'Kitchen/Array/KA5.wav', 'Kitchen/Array/KA6.wav',
             'Kitchen/Wall/K1L.wav',  'Kitchen/Wall/K1R.wav',
             'Kitchen/Wall/K2L.wav',  'Kitchen/Wall/K2R.wav',
             'Kitchen/Wall/K3C.wav',  'Kitchen/Wall/K3L.wav',
             'Kitchen/Wall/K3R.wav',  'Livingroom/Array/LA1.wav',
             'Livingroom/Array/LA2.wav', 'Livingroom/Array/LA3.wav',
             'Livingroom/Array/LA4.wav', 'Livingroom/Array/LA5.wav',
             'Livingroom/Array/LA6.wav', 'Livingroom/Wall/L1C.wav',
             'Livingroom/Wall/L1L.wav',  'Livingroom/Wall/L1R.wav',
             'Livingroom/Wall/L2L.wav',  'Livingroom/Wall/L2R.wav',
             'Livingroom/Wall/L3L.wav',  'Livingroom/Wall/L3R.wav',
             'Livingroom/Wall/L4L.wav',  'Livingroom/Wall/L4R.wav' ]
             
# Ref mics for each room
REF_MICS = { 'BEDROOM':  'Wall/B1L.wav', 'LIVINGROOM': 'Array/LA6.wav', 
             'KITCHEN': 'Array/KA3.wav', 'BATHROOM': 'Wall/R1R.wav', 
             'CORRIDOR' : 'Wall/C1R.wav' } 

# DIRHA CORPUS MATCHING REGEXP
# DIRHA simulated corpora
DIRHA_SIM  = re.compile('(.*)/*DIRHA_sim2/*([A-Z\*]+)/+([a-z0-9\*]+)/+[a-z\*]*/*'
                        'sim([0-9]+)/+Signals/+Mixed_Sources/+([A-Za-z\*]+)/+'
                        '([A-Za-z\*]+)/+([A-Z0-9\*]+)\.([^\.\*]*)$')
# GRID-DIRHA
DIRHA_GRID = re.compile('(.*)/*(grid)_dirha/*([a-z0-9\*]+)/+sim([0-9]+)/+Signals/+'
                        'Mixed_Sources/+([A-Za-z\*]+)/+([A-Za-z\*]+)/+([A-Z0-9\*]+)'
                        '\.([^\.\*]*)$') 
# GRID-DIRHA features
DIRHA_GRID_MFC = re.compile('./features/(DIRHA_sim2/ITA|DIRHA_sim2/PT'
                            '|DIRHA_sim2/GR|DIRHA_sim2/DE|grid_dirha)/'
                            '(dev1|test1|test2)/sim([0-9]*)/Signals/'
                            'detection.mfc') 

# ASR SPEECH EVENT
SPA_EV = re.compile('sp_comm_read|sp_cmd[0-9]+')
# ANY SPEECH EVENT
SP_EV = re.compile('sp_.*')

##############
#  FUNCTIONS 
##############

def comp_DIRHA_path(root, lang, sets, sim, room, device, mic, typ):
    if lang == 'grid':
        if root == '*':
            return ('%s%s_dirha/%s/sim%s/Signals/Mixed_Sources/%s/%s/%s.%s' 
                   % (root, lang, sets, sim, room, device, mic, typ))      
        else:
            return ('%s/%s_dirha/%s/sim%s/Signals/Mixed_Sources/%s/%s/%s.%s' 
                   % (root, lang, sets, sim, room, device, mic, typ))      
    else:  
        if root == '*':
            return ('%sDIRHA_sim2/%s/%s/sim%s/Signals/Mixed_Sources/%s/%s/%s.%s' 
                   % (root, lang, sets, sim, room, device, mic, typ))      
        else:
            return ('%s/DIRHA_sim2/%s/%s/sim%s/Signals/Mixed_Sources/%s/%s/%s.%s' 
                   % (root, lang, sets, sim, room, device, mic, typ))      

def comp_DIRHA_mics_path(root, lang, sets, sim):
    if lang == 'grid':
        if root == '*':
            return ('%s%s_dirha/%s/sim%s/Signals/Mixed_Sources/' 
                    % (root, lang, sets, sim))      
        else:
            return ('%s/%s_dirha/%s/sim%s/Signals/Mixed_Sources/' 
                    % (root, lang, sets, sim))      
    else:  
        if root == '*':
            return ('%sDIRHA_sim2/%s/%s/sim%s/Signals/Mixed_Sources/' 
                    % (root, lang, sets, sim))      
        else:
            return ('%s/DIRHA_sim2/%s/%s/sim%s/Signals/Mixed_Sources/' 
                    % (root, lang, sets, sim))      




def fix_room_name(room):
    '''
    Recover name fo room when cropped
    '''
    room = room.upper()
    if room in ['BAT', 'BATHROOM']:
        room = 'BATHROOM' 
    elif room in ['BEDROO', 'BED', 'BEDROOM']:
        room = 'BEDROOM' 
    elif room in ['CORR', 'COR', 'CORRIDOR']:
        room = 'CORRIDOR' 
    elif room in ['KIT', 'KITCHEN']:
        room = 'KITCHEN' 
    elif room in ['LIV', 'LIVINGROOM']:
        room = 'LIVINGROOM' 
    else:
        raise EnvironmentError, "Unrecognized room name, cropped? %s" % room
    return room

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

    # ADMIT ALSO THE WAV PATH INSTEAD OF TXT
    if re.match('.*\.wav$', txt_path):
        txt_path = re.sub('\.wav$', '.txt', txt_path)

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
            elif line == '<MICROPHONE>':
                state       = 'microphone' 
                mics = {}
            else:
                pass
        # On global state
        elif state == 'global':
            # Exit
            if line == '</GLOBAL>':
                state = None
            else:
                events.append(line.rstrip().split()) 
        # On local state
        elif state == 'source':
            # Exit
            if line == '</SOURCE>':
                # Append dictionary
                sources.append(source)
                state = None
            # Tag found 
            else:
                # tag value /tag format 
                if re.match('\s*<([^>]*)>([^>]*)<.*', line):
                    mt = re.search('\s*<([^>]*)>([^>]*)<.*', line)
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
                    
                    elif mt.group(1) == 'pos':
                        fetch = re.search(' *xs=([0-9\.]+) ys=([0-9\.]+) '
                                          'zs=([0-9\.]+ *) REF=REF_(\w*) *', 
                                          mt.group(2)) 
                        source['pos'] = np.array([int(po) for po in 
                                                  fetch.groups()[:3]])
                        room = fix_room_name(fetch.groups()[3])
                        source['room'] = room[0].upper() + room[1:].lower()
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
        # On global state
        elif state == 'microphone':
            # Exit microphone
            if line == '</MICROPHONE>':
                state = None
            # tag value /tag format 
            elif re.match('\s*<([^>]*)>([^>]*)<.*', line):
                #import ipdb;ipdb.set_trace()
                mt = re.search('\s*<([^>]*)>([^>]*)<.*', line)
                if mt.groups()[0] == 'mic_pos':
                    fetch = re.search(' *x=([0-9\.]+); y=([0-9\.]+); '
                                      'z=([0-9\.]+ *); REF_(\w*) *', 
                                      mt.groups()[1]) 
                    mics['pos'] = np.array([float(po) for po in 
                                            fetch.groups()[:3]])
                    room         = fix_room_name(fetch.groups()[3])
                    mics['room'] = room[0].upper() + room[1:].lower()
                else:
                    mics[mt.groups()[0]] = mt.groups()[1]
            else:
                raise ValueError, ("Could not parse MICROPHONE line:\n\n%s"
                                    % line) 
        # Read 
        i += 1

    # SANITY CHECK: WE READ SOMETHING
    if not len(sources):
        raise EnvironmentError, ("File %s does not seem to be a DIRHA"
                                 " meta-data file" % (txt_path))

    return (events, sources, mics)


def dsmp(T, in_fs, work_fs):
    '''
    Adjusts time measured in samples to match the working sampling 
    frequency 
    '''
    return int(np.floor(T*work_fs*1.0/in_fs))

##############
#  CLASSES 
##############

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
        self.path = {}
        if DIRHA_SIM.match(txt_path): 
            in_fs = 48000
            # EXTRACT CORPUS CHARACTERISTICS FROM METADATA FILE NAME
            [self.path['root'], self.path['lang'], self.path['sset'], sim, 
             self.path['room'], self.path['device'], self.path['mic'], 
             self.path['ftype']] = DIRHA_SIM.search(txt_path).groups() 
            self.path['sim'] = int(sim)
     
        elif DIRHA_GRID.match(txt_path): 
            in_fs = 16000 
            # EXTRACT CORPUS CHARACTERISTICS FROM METADATA FILE NAME
            [self.path['root'], self.path['lang'], self.path['sset'], sim, 
             self.path['room'], self.path['device'], self.path['mic'], 
             self.path['ftype']] = DIRHA_GRID.search(txt_path).groups() 
            self.sim = int(sim)
     
        else: 
            raise EnvironmentError, ("%s does not seem to be a DIRHA corpus"
                                     " file" % txt_path)

        # Extract data from the file
        self.glob, self.sources, self.mics = readmetadata(txt_path, 
                                                          in_fs=in_fs, 
                                                          work_fs=work_fs)


    def get_mics_path(self):

        return comp_DIRHA_mics_path(self.path['root'], self.path['lang'], 
                                    self.path['sset'], self.sim)

    def get_ref_mic_source(self, src):
        '''
        Returns the reference microphone in the room were this src took 
        place for this simulation.
        ''' 
        room = fix_room_name(src['room'])
        return (self.txt_path.split('/Mixed_Sources/')[0] 
                + '/Mixed_Sources/' + room[0] + room[1:].lower() 
                +  '/' + REF_MICS[room]) 

    def get_sp_list(self):
        '''
        Return speech sources
        ''' 
        sp_list = []
        for s in self.sources: 
            if SP_EV.match(s['name']):
                sp_list.append(s['name'])
        return sp_list

    def get_sp_dict(self):
        '''
        Return speech sources
        ''' 
        sp_dict = {}
        for s in self.sources: 
            if SP_EV.match(s['name']):
                sp_dict[s['name']] = s
        return sp_dict

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
