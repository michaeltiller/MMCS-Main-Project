import os, datetime
def path( *args, **kwargs):
     """returns an OS independent path out of its arguments"""
     return os.path.join(*args, **kwargs)

def timestamp():
     """returns a string timestamp of the day and hour right now"""
     return datetime.datetime.now().strftime('%d %b %Y %I-%M%p')
