import sys


class ConversionError(Exception):
    '''
    an abtract class
    '''
    pass


def print_stderr(msg):
    '''
    a function to print information to the std
    '''
    sys.stderr.write('%s\n' % msg)
