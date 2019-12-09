import importlib 
import sys
    
class Reloader:
    def __init__(self, files):
        self.files = files 
    
    def reload(self) : 
        for f in self.files : 
            importlib.reload(sys.modules[f]) 
            print("Reloaded: {}".format(f)) 
            
            
            

