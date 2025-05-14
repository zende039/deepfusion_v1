import platform
import getpass
import os

EXP_DIR = 'HO_Duration/'
RAWDATA_DIR = 'Data/'
PROCESSDATA_DIR = 'ProcessedData/'


if platform.system() == 'Linux' and getpass.getuser() == 'umn-networking':
    # rushikesh - laptop
    EXPR_FOLDER = '/home/umn-networking/Documents/{}'.format(EXP_DIR)
    RAWDATA_FOLDER = '/home/umn-networking/Documents/{}{}'.format(EXP_DIR, RAWDATA_DIR)
    PROCESSDATA_FOLDER = '/home/umn-networking/Documents/{}{}'.format(EXP_DIR,PROCESSDATA_DIR )
elif platform.system() == 'Linux' and getpass.getuser() == 'rushi':
    # rushikesh - GPU Linux Desktop
    EXPR_FOLDER = '/home/rushi/Documents/'
    RESULTS_FOLDER = '/home/rushi/Documents/'
    RAWDATA_FOLDER = '/home/rushi/Documents/{}'.format(RAWDATA_DIR)
    PROCESSDATA_FOLDER = '/home/rushi/Documents/{}'.format(PROCESSDATA_DIR)
elif platform.system() == 'Linux' and getpass.getuser() == 'zzadmin':
    # rushikesh - GPU Linux Desktop
    EXPR_FOLDER = '/home/zzadmin/Documents/'
    RESULTS_FOLDER = '/home/zzadmin/Documents/'
    RAWDATA_FOLDER = '/home/zzadmin/Documents/{}'.format(RAWDATA_DIR)
    PROCESSDATA_FOLDER = '/home/zzadmin/Documents/{}'.format(PROCESSDATA_DIR)
elif platform.system() == 'Darwin' and getpass.getuser() == 'rushikeshzende':
    # rushikesh - Desktop
    RAWDATA_FOLDER = '/Users/rushikeshzende/Google Drive/Shared drives/{}'.format(RAWDATA_DIR)



