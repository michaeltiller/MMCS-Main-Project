import os, datetime
def path( *args, **kwargs):
     return os.path.join(*args, **kwargs)

def timestamp():
     return datetime.datetime.now().strftime('%d %b %Y %I-%M%p')
