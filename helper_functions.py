import os, datetime
def path( *args, **kwargs):
     """returns an OS independent path out of its arguments"""
     return os.path.join(*args, **kwargs)

def timestamp():
     """returns a string timestamp of the day and hour right now"""
     return datetime.datetime.now().strftime('%d %b %Y %I-%M%p')

def get_MIP_gap(prob):
     best_obj = prob.attributes.objval
     best_bound = prob.attributes.bestbound

     if best_obj != 0:
          mip_gap = abs(best_obj - best_bound) / abs(best_obj)
     else:
          mip_gap = float('inf')
     return mip_gap

