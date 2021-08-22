import mpi4py
from tqdm.auto import tqdm

__all__ = ['isnotebook', 'ptqdm']

def isnotebook():
    """
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


class ptqdm:
    """ptqdm for parallel use of tqdm"""
    
    __config__ = {'ncols': None if isnotebook() else 80, 'ascii': True}
    
    def __init__(self, *args, **kwargs):
        
        comm = kwargs['comm'] if 'comm' in kwargs.keys() else None
        comm = comm or mpi4py.MPI.COMM_WORLD
        self.rank = comm.Get_rank()  
        
        for key, val in ptqdm.__config__.items():
            if key not in kwargs.keys():
                kwargs[key] = val
        
        self.tqdm = tqdm(*args, **kwargs) if self.rank == 0 else None
    
    def update(self):
        if self.tqdm is not None:
            self.tqdm.update()
            
    def close(self):
        if self.tqdm is not None:
            self.tqdm.close()
        
    def __getattr__(self, attr):
        return self.tqdm.__get_attr__(attr)