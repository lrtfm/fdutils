import mpi4py
from tqdm.auto import tqdm

__all__ = ['isnotebook', 'ptqdm']

def isnotebook():
    """
    Check whether you are in notebook.
    
    Ref:
        url: https://stackoverflow.com/a/39662359
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


class ptqdm(tqdm):
    """A parallel wraper of tqdm (progressbar)"""
    
    __config__ = {'ncols': None if isnotebook() else 80, 'ascii': True}
    
    def __init__(self, *args, **kwargs):
        
        comm = kwargs['comm'] if 'comm' in kwargs.keys() else None
        comm = comm or mpi4py.MPI.COMM_WORLD
        self.rank = comm.Get_rank()  
        
        for key, val in ptqdm.__config__.items():
            if key not in kwargs.keys():
                kwargs[key] = val

        if self.rank != 0:
            kwargs['disable'] = True
        
        super(ptqdm, self).__init__(*args, **kwargs)
