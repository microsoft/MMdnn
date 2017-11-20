import re

    
def get_lower_case(text):
    '''
    Convert PascalCase name to words concatenated by '_'.
    'PascalCase' -> 'pascal_case'
    '''
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

    
def get_upper_case(text):
    '''
    'pascal_case' -> 'PascalCase'
    '''
    return ''.join([item.title() for item in text.split('_')])

def get_real_name(text):
    text = text.strip().split(':')
    return ''.join(text[:-1])

def listToStr(data):
    ret = ""
    first = True
    for e in data:
        if first == False:
            ret += ", "
        ret += str(e)
        first = False
    return ret