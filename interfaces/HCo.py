#!/usr/bin/python
'''

Python version of HTK HCopy executable that calls Python code to perform
feature extraction.   

Input: HCopy_UP_FOLDER   string with path to the folder where the MCopy
                         tools are, typically ./custom_fe/MAT/htk/

       HCopy_args        string with the arguments of a standard HTK call
                          
In HCopy_args a configuration file must always be specified with -C
this configuration must contain the field CUSTOM_FEATS_FOLDER pointing
to a folder where the feature extraction is defined by two functions.
Another alternative is to set CFF_FROM_CONFIG_PATH = T. In this case the 
folder path of the current config file will be set as CUSTOM_FEATS_FOLDER
variable. 

Use custom_fe/PY/custom/IS2014/ as an example of feature extraction

MCopy supports HTK -C and -S options only as well as a single source
and target pairs

Three non-HTK options are also provided

-resume   If file exists skip it        
-debug    When launching MCopy it inserts a breakpoint at the beginning
-up       It signals the feature extraction to append the variance of
          the features by setting config field UNC_PROP to 1

'''

import sys
import os

# Ramon's custom toolboxes
# Functions that are called as scripts need to be able to load the toolbox as
# it is not installed. 
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import interfaces.htk as htk 

####################
#     FUNCTIONS
####################

def in_and_eq(config, field, value):
    '''
    Returns True if field is set and equals given value 
    '''
    if field not in config:
        return False
    elif config[field] == value:
        return True
    else:
        return False

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


def parse_HCopy_args(HCopy_args, HTK_cf=None):
    '''
    Parses the arguments to HTKs HCopy function and returns them as variables 
    usable by Python
    '''
    # If not given intitialize HTK_cf
    if not HTK_cf:
        HTK_cf = {}
    # Control flags
    no_config = 1
    no_scp    = 1
    no_files  = 1
    #
    argi      = 1
    while argi < len(HCopy_args):
        # NON HTK OPTIONS
        # Resume mode 
        if HCopy_args[argi] == "-resume":
            HTK_cf['do_resume'] = 1
        # Debug mode 
        elif HCopy_args[argi] == "-debug":
            HTK_cf['do_debug'] = 1
        # Uncertainty Propagation mode
        elif HCopy_args[argi] == "-up":
            HTK_cf['do_up'] = 1
        # HTK OPTIONS
        # Argumentless options
        elif HCopy_args[argi] in ["-D", "-A"]:
            # Display configuration settings 
            pass
        # Version mode
        elif HCopy_args[argi] == "-V":
            HTK_cf['show_version'] = 1
            src_files               = []
            trg_files               = []
        # Trace
        elif HCopy_args[argi] == "-T":
            # NOTE: This is ignored right now
            T     = int(HCopy_args[argi+1])
            argi += 1
        # Not supported
        elif HCopy_args[argi] in ['-a', '-e', '-argi', '-l', '-m', '-n', '-s',
                                  '-t', '-x', '-F', '-G', '-I', '-L', '-O', 
                                  '-P', '-X']:
            # No implemented yet
            raise NotImplementedError, ("Option %s not supported"
                                        " yet by HCo.py") % HCopy_args[argi]   
        # Read configuration 
        elif HCopy_args[argi] == "-C":
            HTK_cf = htk.readhtkconfig(HCopy_args[argi+1], HTK_cf)
            no_config  = 0
            argi      += 1
    
        elif HCopy_args[argi] == "-S":
            [src_files, trg_files] = htk.readscp(HCopy_args[argi+1])
            no_scp                       = 0
            argi                       += 1
        # Assume this is a target file
        else:
            if not os.path.isfile(HCopy_args[argi]):
                raise OSError, ("You provided an optionless argument %s\nthi"
                                "s has to be an existing file" 
                                % HCopy_args[argi])
            if not no_scp:
                raise OSError, ("You provided both a script file with -S"
                               " and a file to open %s") % HCopy_args[argi]
    
            if argi+2 > len(HCopy_args):
                src_files = [HCopy_args[argi]]
                trg_files = [None]
    
            else:
                src_files = [HCopy_args[argi]]
                trg_files = [HCopy_args[argi+1]]
            no_files = 0
            argi    += 1
        argi += 1 

    # SANITY CHECKS
    # At least one config given
    if no_config:
        raise ValueError, ("You need to specifiy a config file with "
                           "-C containing parameter custom_feats_folder")
    # Feature extraction path specified
    if 'custom_feats_folder' not in HTK_cf:
        raise ValueError, ("You need to specifiy parameter custom_feats_folder"
                           " on the config files")
    # Feature extraction path exists
    if not os.path.exists(HTK_cf['custom_feats_folder']):
        raise OSError, ("The feature extraction code could not"
                        " be found on %s") % HTK_cf['CUSTOM_FEATS_FOLDER']
    # Check for valid SCP file
    # File list or files provided
    if no_files and no_scp:
        raise ValueError, "No list of files provided with -S or input files"
    # It is a HCopy type file
    if no_scp == 0 and trg_files == []:
        raise ValueError, "You provided an scp file with no targets"
    
    return [src_files, trg_files, HTK_cf]

##########################
#         MAIN
##########################

if __name__ == '__main__':

    # PARSE HTK CALL, DO SOME BASIC CHECKS, COMPLETE DEFAULTS	
    [source_files, target_files, HTK_config] = parse_HCopy_args(sys.argv)

    # REMOVE FLAGS INTENDED FOR HCO.PY FROM THE CONFIG
    do_resume = 0
    for fieldname in HTK_config.keys:
        if fieldname == 'do_resume':
            do_resume = HTK_config['do_resume']
            HTK_config.pop('do_resume')

    # INITIALIZE FEATURE EXTRACTION 
    if 'custom_feats_folder' in HTK_config:
        sys.path.append(HTK_config['custom_feats_folder'])
        import custom_feat
        # Initialize
        features = custom_feat.FRONTEND(HTK_config['custom_feats_folder'], 
                                        HTK_config)
    else:
        raise EnvironmentError, ("You need to specify CUSTOM_FEATS_FOLDER "
                                 "variable in the config")

    # CHECK FOR VALID CONFIG
    features.validate_config()

    # RUN FEATURE EXTRACTION ON BATCH MODE 
    n_files = len(source_files)
    print "Extracting %d file(s)" % n_files 
    for i in range(0, n_files):
        # INFO
        sys.stdout.write("\rFile %d/%d %s -> %s" % (i+1, n_files, 
                         source_files[i], target_files[i]))
        sys.stdout.flush()

        # SANITY CHECK: File exists 
        if not os.path.exists(source_files[i]):
            raise OSError, ("Source file %s does not exist!! " 
                            % (source_files[i]))
        # Skip if target exists and resume mode on
        if do_resume and os.path.exists(target_files[i]):
            continue    
        # Create folder of target if it does not exist
        if target_files[i] != []:
            target_folder = os.path.dirname(target_files[i])
            if target_folder and not os.path.isdir(target_folder):
                try:    
                    os.makedirs(target_folder)
                except OSError:    
                    print "Folder %s could not be created" % target_folder 
    
        # Read audio
        features.read(source_files[i])
        # Extract features 
        features.extract()    
        # Write features
        features.write(target_files[i])
        print ""
