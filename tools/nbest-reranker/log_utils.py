import logging
import sys

#-----------------------------------------------------------------------------------------------------------#
import re

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'
    BHEADER = BOLD + '\033[95m'
    BOKBLUE = BOLD + '\033[94m'
    BOKGREEN = BOLD + '\033[92m'
    BWARNING = BOLD + '\033[93m'
    BFAIL = BOLD + '\033[91m'
    BUNDERLINE = BOLD + '\033[4m'
    BWHITE = BOLD + '\033[37m'
    BYELLOW = BOLD + '\033[33m'
    BGREEN = BOLD + '\033[32m'
    BBLUE = BOLD + '\033[34m'
    BCYAN = BOLD + '\033[36m'
    BRED = BOLD + '\033[31m'
    BMAGENTA = BOLD + '\033[35m'
    BBLACK = BOLD + '\033[30m'

    @staticmethod
    def cleared(s):
        return re.sub("\033\[[0-9][0-9]?m", "", s)

def red(message):
    return BColors.RED + str(message) + BColors.ENDC

def b_red(message):
    return BColors.BRED + str(message) + BColors.ENDC

def blue(message):
    return BColors.BLUE + str(message) + BColors.ENDC

def yellow(message):
    return BColors.YELLOW + str(message) + BColors.ENDC

def b_yellow(message):
    return BColors.BYELLOW + str(message) + BColors.ENDC

def white(message):
    return BColors.WHITE + str(message) + BColors.ENDC

def green(message):
    return BColors.GREEN + str(message) + BColors.ENDC

def b_green(message):
    return BColors.BGREEN + str(message) + BColors.ENDC

def b_okblue(message):
    return BColors.OKBLUE + str(message) + BColors.ENDC

def b_fail(message):
    return BColors.BFAIL + str(message) + BColors.ENDC

def b_warning(message):
    return BColors.WARNING + str(message) + BColors.ENDC

def print_args(args, path=None):
    if path:
        output_file = open(path, 'w')
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    args.command = ' '.join(sys.argv)
    items = vars(args)
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        logger.info("  " + key + ": " + str(items[key]))
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    if path:
        output_file.close()
    del args.command

#-----------------------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------------------#

def set_logger(out_dir=None, log_file="log.txt"):
    #console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + ' (%(name)s) %(message)s'
    #console_format = b_okblue('[%(levelname)s]') + b_okblue(' [%(asctime)s] ') + ' %(message)s '
    datefmt='%d-%m-%Y %H:%M:%S'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(ColoredFormatter(datefmt=datefmt))
    logger.addHandler(console)
    if out_dir:
        #file_format = '[%(levelname)s] (%(name)s) %(message)s'
        file_format = '[%(levelname)s] [%(asctime)s] %(message)s'
        log_file = logging.FileHandler(out_dir + '/' + log_file, mode='w')
        log_file.setLevel(logging.DEBUG)
        log_file.setFormatter(logging.Formatter(file_format, datefmt=datefmt))
        logger.addHandler(log_file)

#-----------------------------------------------------------------------------------------------------------#

class ColoredFormatter(logging.Formatter):
    FORMATS = {logging.DEBUG :"DBG: %(module)s: %(lineno)d: %(message)s",
               logging.ERROR : b_fail('[%(levelname)s]') + ' [%(asctime)s] ' + ' %(message)s ',
               logging.INFO : b_okblue('[%(levelname)s]') + ' [%(asctime)s] ' + ' %(message)s ',
               logging.WARNING : b_warning('[%(levelname)s]') + ' %(message)s',
               'DEFAULT' : b_okblue('[%(levelname)s]') + ' [%(asctime)s] ' + ' %(message)s '}

    def format(self, record):
        self._fmt = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        return logging.Formatter.format(self, record)
