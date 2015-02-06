#!/usr/bin/python
'''
Interface functions to HTK formats. For reference see the HTKbook 3.4

Ramon F. Astudillo
'''

import os
import re
import struct
import numpy as np

def targetkind2num(targetkind):
    '''
    Computes binary representation of TARGETKIND in HTK
    '''
    # TYPES AND MODIFIERS ALLOWED IN HTK
    types = ['WAVEFORM', 'LPC', 'LPREFC', 'LPCEPSTRA', 'LPDELCEP', 'IREFC', 
             'MFCC', 'FBANK', 'MELSPEC', 'USER', 'DISCRETE', 'PLP', 'ANON']
    mods  = ['E', 'N', 'D', 'A', 'C', 'Z', 'K', '0', 'V', 'T']
    # TYPE AND MODIFIERS OF THE FEATURE VECTOR
    tokens = targetkind.split('_')
    if tokens[0] not in types:
        raise ValueError, "Unknown TARGETKIND type %s" % tokens[0]
    # COMPUTE HTK FORMAT
    # type
    htk_format = types.index(tokens[0])
    # Modifiers
    for mod in tokens[1:]:
        if mod not in mods:
            raise ValueError, "Unknown TARGETKIND modifier %s" % mod
        htk_format += 2**(6 + mods.index(mod)) 
    return htk_format 

def nextpow2(N):
    '''
    Raise N to the nex power of 2 
    '''
    n = 1
    while n<N: n *= 2
    return int(n)

def readhtkresult(file_path, stdout=False):
    '''
    CHAPUZA VERSION
    '''
    with open(file_path) as f:
        lines = f.readlines() 
    # Expected format
    if lines[17].split()[1]  == 'Sum/Avg':
        WER= lines[17].split()[9] 
    else:
        raise IOError, "Unknown format in htk result %s" % file_path
    if stdout:
        print WER 
    else:
        return WER

def readhtkconfig(file_path, config, htk_units=False):
    '''
    Reads HTK style config into a dictionary 

    Input: file_path  string with path to a HTK config file
    Input: prev_dict  dictionary with previously read fields   
    '''
    with open(file_path) as f:
        for line in f.readlines():
            if not (re.match('^\s*#.*$', line) or re.match('^\s*$', line)):
                
                line     = line.split('#')[0].strip()
                keyname  = line.split('=')[0].strip().lower()
                keyvalue = line.split('=')[1].strip()

                # If true/false, convert to logical  
                if keyvalue == 'T':
                     config[keyname] = True 
                elif keyvalue == 'F':    
                     config[keyname] = False 
                # Try to convert to number
                elif keyvalue.isdigit():  
                     config[keyname] = int(keyvalue)
                else:
                    try:
                         config[keyname] = float(keyvalue)
                    except ValueError:
                         config[keyname] = keyvalue

                # Special case, extract custom_feats_folder from config path
                if keyname == 'cff_from_config_path' and keyvalue == 'T':
                    config['custom_feats_folder'] = os.path.dirname(file_path)
 
    # Translate HTK units (1e-7 secs) to normal units (samples, secs), also
    # complete basic STFT config
    if not htk_units:
        # SAMPLING FREQUENCY  
        if 'sourcerate' in config:
            config['work_fs'] = int(1e7/config['sourcerate'])
            config.pop('sourcerate') 
        elif 'work_fs' not in config:
            raise ValueError, "You need either to specifiy SOURCERATE or work_fs"

        # WINDOWSIZE AND SHIFT 
        if ('targetrate' in config) and ('windowsize' in config):
            config['shift']      = int(config['work_fs']*
                                       config['targetrate']*1e-7) 
            config['windowsize'] = int(config['work_fs']*
                                       config['windowsize']*1e-7)
            # Sanity check: Maybe windowsize was defined in samples
            if config['windowsize'] == 0:
                raise ValueError, ("windowsize seems to be given in samples HTK"
                                   " units expected")
            config['nfft']       = nextpow2(config['windowsize']) 
            config.pop('targetrate') 
        elif not (('shift' in config) and ('windowsize' in config)):
            raise ValueError, ("Either targetrate and windowsize in HTK units or"
                               "shift and windowsize in samples need to be defined")
        elif 'nfft' not in config:
            config['nfft'] = nextpow2(config['windowsize'])


    return config

def readhtkfeats(htkfeats_file):
    '''
    Reads a matrix of feature vectors written with the Hidden Markov Model
    Toolbox (HTK) format. Output variable name follows conventions of the
    VOICEBOX toolbox for MATLAB see

    http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

    Ramon F. Astudillo
    '''

    # SANITY CHECK: FILE EXISTS
    if not os.path.isfile(htkfeats_file):
        raise IOError, "File %s does not exist" % htkfeats_file

    # Open file
    with open(htkfeats_file, 'r') as fid:
        # READ HEADER
        [L, fp, by, tc]  = struct.unpack('>LLhh', fid.read(12))
        # Convert HTK units to seg
        fp             = fp*1e-7
        # DETERMINE DATA TYPE (lower six bits of TC)
        dt = sum([(2**i)*((tc>>i)&1) for i in xrange(5, -1, -1)])
        # SANITY CHECK: UNSUPORTED
        if dt == 0 or dt == 5 or dt == 10:
            raise NotImplementedError, "Sorry 16bit data not supported"
        # READ REST
        x = np.array(struct.unpack('>'+'f'*(by/4)*L, fid.read(by*L)))
        x = np.reshape(x, (by/4, L), order = 'F')

        return [x, fp, dt, tc]


def writehtkfeats(htkfeats_file, x, fp, tc):
    '''
    Writes a matrix of feature vectors with the Hidden Markov Model Toolbox
    (HTK) format. Output variable name follows conventions of the VOICEBOX
    toolbox for MATLAB see

    http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

    Input: htkfeats_file string   with path to the file to be written

    Input: x             [I, L]   ndarray of I features and L frames
                                  NOTE that this is the transpose of the 
                                  format used in the VOICEBOX toolbox

    Input  fp            float    frame period in seconds

    Input: tc            int      HTK targetkind in integer form              
    '''

    # Get sizes
    [I, L] = x.shape
    # SANITY CHECK: We are not writing matrix transposed
    if 4*I > 32767:
        raise ValueError, ("Length of feature vectors to big, is matrix"
                          "tranposed?")
    # SANITY CHECK: FOLDER EXISTS
    htkfeats_folder = os.path.dirname(htkfeats_file)
    if htkfeats_folder != '' and not os.path.isdir(htkfeats_folder):
        raise IOError, "Folder %s does not exist" % htkfeats_folder
    # Die if CRC (first bit of tc) solicited
    if tc >> 31:
        raise NotImplementedError, "CRC not supported, check tc value"
    # Write file
    with open(htkfeats_file, 'w') as fid:
        # WRITE HEADER
        fid.write(struct.pack('>LLhh', L, round(fp*1.E7), 4*I, tc))
        # WRITE BODY
        fid.write(struct.pack('>'+'f'*I*L, *(x.T.ravel())))


def readmlf(mlf_path):
    '''
    Reads an HTKs master label file returning a list of tuples containing
    filename and the transcription
    '''
    # SANITY CHECK, paths exists
    if not os.path.exists(mlf_path):
        raise IOError, "ERROR path of MLF file  %s dos not exist" % (mlf_path)
    # Open file safely
    with open(mlf_path) as fid_mlf:
        mlf     = []          
        n_line  = 0
        in_sent = 0
        names   = []  
        for line in fid_mlf.readlines():
            # Skip comments
            if re.match('\s*#.*', line):
                continue
            n_line += 1
            # In sentence transcription. If single '.' found transcription ends
            if in_sent:
                if re.match('\.\n', line):
                    mlf.append(sent_entry) 
                    in_sent = 0
                else:
                    # Extract rich transcription
                    #
                    # [start [end] ] name [score] { auxname [auxscore] }
                    # [comment]
                    #
                    items = line.rstrip().split()
                    # Only word name given
                    if len(items) == 1:
                        sent_entry[1].append([-1, -1, items[0]])
                    # Start, End, state, [name, auxname], last two missing
                    # after start word
                    elif len(items) == 3 and names != []:
                        #sent_entry[1].append(items[:] + names)
                        sent_entry[1].append(items[:])
                    # Start, End and name
                    # NOTE: this could also be "start name auxname/comment"
                    # by we assume this is very unlikely
                    elif len(items) == 3:
                        sent_entry[1].append(items[:])
                    elif len(items) == 4: 
                        # Start, End, name, score
                        if re.match('-?[0-9]+', items[3]):
                            sent_entry[1].append(items[:])
                        # Start, End, state, name
                        else:
                            #sent_entry[1].append(items[:] + [names[1]])
                            sent_entry[1].append(items[:])
                            names[0] = items[3]
                    # Start, End, state, name, auxname
                    elif len(items) == 5:
                        sent_entry[1].append(items[:])
                        names = items[3:]
                    else:
                        raise ValueError, ("Unknown transcription format in"
                                           "line %d of"
                                           "%s") % (n_line, mlf_path)
            # In sentence name
            elif re.match('^\"[^\"]+\.\'*\w+\'*\"$', line):
                # Create a new sentence entry that will be appended to the mlf
                # list when we exit the transcription
                sent_entry = [line.rstrip().replace("\"", ""), []]
                in_sent    = 1
            else:
                raise ValueError, (("Missing sentence start at line %d of %s") 
                                   % (n_line, mlf_path))
    # Return separately
    return mlf 


def writemlf(mlf, mlf_path, file_term='lab'):
    '''
    writes mlf from a list 
    '''
    # SANITY CHECK, folder where file must be written exists
    target_path = os.path.split(mlf_path)[0]
    if target_path != '' and not os.path.exists(target_path):
        raise IOError, ("ERROR folder where to create MLF file"
                         "%s does not exist") % (os.path.split(mlf_path)[0])
    fid = open(mlf_path, 'w')
    # For each utterance stored
    fid.write('#!MLF!#\n')
    for sent in mlf:
        fid.write('\"%s\"\n' % (".".join(sent[0].split('.')[:-1]) + '.' 
                  + file_term))
        auxname = ''
        name    = ''
        for word in sent[1]:
            fid.write(" ".join(word[:2] + [str(word[2])]  + word[3:]) + '\n')
        fid.write('.\n')

def readmlf2dict(mlf_path, keytype = 'filename'):
    '''
    READMLF2DICT: Reads an HTKs master label file returning a dictionary
    indexed by filename. Each dictionary entry contains a list with each
    segment transcription

    Set HTKbook 3.4, page 87


    Input keytype determines the keys used in the dict: 


             'filename'   basename used 
             'filepath'   whole path is used

    '''
    # SANITY CHECK, paths exists
    if not os.path.exists(mlf_path):
        raise IOError, "ERROR path of MLF file  %s dos not exist" % (mlf_path)

    # Open file safely
    with open(mlf_path) as fid_mlf:

        # Atempt to extract the file content as chunks of filename followed by
        # rich transcription (at least labels, eventually timestamps and
        # log-likelihoods)

        # THIS DOES NOT WORK AS SOME WORDS CONTAIN THE TOKEN "."
  # att_match = re.findall('\"([^\"]*)\.rec\"\n([^\.]+\n)\.\n', fid_mlf.read())

        # Initialize dictionary
        trans_dict = {}

        n_line = 0
        # State
        in_sent = 0
        names = []  # If state numbers provided, the name and auxname are
                  # appended to all transcriptions (not just the first)
        # For each line
        for line in fid_mlf.readlines():

            # Skip comments
            if re.match('\s*#.*', line):
                continue

            n_line += 1
            if in_sent:
                # If single '.' found, sentence ended
                if re.match('\.\n', line):
                    in_sent = 0
                else:

                    # Extract rich transcription
                    #
                    # [start [end] ] name [score] { auxname [auxscore] }
                    # [comment]
                    #
                    items = line.rstrip().split()
                    # Only word name given
                    if len(items) == 1:
                        trans_dict[key_name].append([-1, -1, items[0]])
                    # Start, End, state, [name, auxname], last two missing
                    # after start word
                    elif len(items) == 3 and names != []:
                        trans_dict[key_name].append(items[:] + names)
                    # Start, End and name
                    # NOTE: this could also be "start name auxname/comment"
                    # by we assume this is very unlikely
                    elif len(items) == 3:
                        trans_dict[key_name].append(items[:])
                    elif len(items) == 4: 
                        # Start, End, name, score
                        if re.match('-?[0-9]+', items[3]):
                            trans_dict[key_name].append(items[:])
                        # Start, End, state, name
                        else:
                            trans_dict[key_name].append(items[:] + [names[1]])
                            names[0] = items[3]
                    # Start, End, state, name, auxname
                    elif len(items) == 5:
                        trans_dict[key_name].append(items[:])
                        names = items[3:]
                    else:
                        raise ValueError, ("Unknown transcription format in"
                                           "line %d of"
                                           "%s") % (n_line, mlf_path)

            else:
                # Sentence transcription starts
                if re.match('^\"[^\"]+\.\'*\w+\'*\"$', line):
                    file_name = line.rstrip().replace("\"", "")
                    if keytype == 'filename':
                        key_name = re.sub('\.[^\.]*$', '',
                                          os.path.split(file_name)[1])
                    elif keytype == 'filepath':
                        pieces = os.path.split(file_name)
                        key_name = ( pieces[0] + '/'
                                    + pieces[1].split('.')[0])
                    trans_dict[key_name] = []
                    in_sent = 1
                else:
                    raise ValueError, ("Missing sentence start at"
                                       "line %d of %s") % (n_line, mlf_path)

    # Return separately
    return trans_dict

def writemlf_fromdict(trans_dict, mlf_path, file_term='lab'):
    '''
    Write an MLF file from a dictionary read with readmlf2dict


    Input:  trans_dict   readmlf2dict type of dictionary
    
    Input:  mlf_path     string containing path to the MLF
 
    Input:  file_term    optional file type on each file name 
    '''
    # SANITY CHECK, folder where file must be written exists
    target_path = os.path.split(mlf_path)[0]
    if target_path != '' and not os.path.exists(target_path):
        raise IOError, ("ERROR folder where to create MLF file"
                         "%s dos not exist") % (os.path.split(mlf_path)[0])

    fid = open(mlf_path, 'w')
    # For each utterance stored
    fid.write('#!MLF!#\n')
    for key in trans_dict.keys():
         
        filename = os.path.basename(key)
        if re.match('.*\..*', filename):
            fid.write('\"%s\"\n' % (".".join(key.split('.')[:-1]) + '.' 
                      + file_term))
        else:
            fid.write('\"%s\"\n' % (key + '.' + file_term))

        auxname = ''
        name    = ''
        for word in trans_dict[key]:
            # If there are empty elements
            # Start, End, state, name, auxname
            if len(word) == 5:
                new_auxname = word[4]
                new_name    = word[3]
                # New word
                if new_auxname != auxname:
                    fid.write(" ".join(word[:2] + [str(word[2])]  + word[3:]))
                    auxname = word[4]
                    name    = word[3]
                # New triphone
                elif new_name != name:
                    fid.write(" ".join(word[:2] + [str(word[2])]  + word[3:4]))
                    name    = word[3]
                else:
                    fid.write(" ".join(word[:2] + [str(word[2])]))
            else:
                fid.write(" ".join(word[:2] + [str(word[2])]  + word[3:]))
            fid.write('\n')
        fid.write('.\n')


def mlf_reg2key(token, mlf_dict, strict=False, unique=True):
    '''
    Sees if sentecne matches a regular expression of paths in a MLF
    ''' 
    if unique:
        for pathregxp in mlf_dict.keys():
            if re.match(pathregxp.replace('*','.*'), token):
                return pathregxp 
        if strict:
            raise EnvironmentError, ("An MLF was provided but it has no "
                                     "transcription for %s" % token)
        else:
            return None     
    else:
        matches = []
        for pathregxp in mlf_dict.keys():
            if re.match(pathregxp.replace('*','.*'), token):
                matches.append(pathregxp)

        if strict and not len(matches):
            raise EnvironmentError, ("An MLF was provided but it has no "
                                     "transcriptions for %s" % token)

        elif not len(matches):
            return None     
 
        return matches


def readscp(scp_file, append_source=''):
    '''
    Read scp file. Append append_source to the source_file
    '''
    # SANITY CHECK FILE EXISTS
    if not os.path.isfile(scp_file):
        raise SystemError, "File %s does not exist" % scp_file

    # Get lines
    line_list = open(scp_file).readlines()

    # If line split into two, this is a HCopy type file
    if len(line_list[0].split()) == 2:
        source_list = [(append_source + line.rstrip().split()[0]) 
                       for line in line_list]
        target_list = [line.rstrip().split()[1] for line in line_list]

    # If no target pattern provided, just return source
    else:
        source_list = [(append_source + line.rstrip()) for line in line_list]
        target_list = None

    return [source_list, target_list]


def writescp(scp_file, file_list):
    '''
    Writes a list of files into a scp. If a list of two lists if given it
    interprets the first as sources and the second as targets
    ''' 
    with open(scp_file, 'w') as f:
        # Single list of paths
        if isinstance(file_list[0], str):
            for ffile in file_list:
                f.write('%s\n' % ffile)
        # Source/Target pairs
        else:
            for n, ffile in enumerate(file_list[0]):
                f.write('%s\t%s\n' % (ffile, file_list[1][n]))
