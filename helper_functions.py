import os, datetime
import numpy as np
import matplotlib.pyplot as plt
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

def create_colours_for_locs_and_assignments(locs,assign_labs, cmap_name = "cool"):
    """
    This is for colour assignment when there are certain number of locations ``locs`` 
    and many other points have been assigned to a unique location, as recorded in ``assign_labs``.

    E.g two locations and five points a possible input is assign_labs=[0,1,1,0,0] and possible output is 
    ["blue", "green"], ["blue","green","green","blue","blue"]

    Locations with no points assigned are coloured in gray
    """
    locs_with_assignments = np.unique(assign_labs)
    # will need to do something here when number of locs is large
    # the colours will differ by increasingly smpall degrees
    # an alternative is to limit the size of the cmap to say 20 and shuffle the indexes before being passed
    # so location 5 wont be mapped to index 4 anymore and wont get a colour similar to location 6
    cmap = plt.colormaps[cmap_name].resampled( len(locs_with_assignments) )
    cols_for_locs_with_demand  = cmap(np.arange(len(locs_with_assignments)))


    cols_for_locs = np.zeros((len(locs),4))
    # locations with no demand/desire are shown in grey
    cols_for_locs[:,:] = .5

    cols_for_locs[locs_with_assignments,:] = cols_for_locs_with_demand

    cols_for_points = cols_for_locs[locs_with_assignments]

    return cols_for_locs, cols_for_points