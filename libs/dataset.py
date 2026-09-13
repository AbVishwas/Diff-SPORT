import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset

def read_hdf5(data_file):
    import h5py
    data_arr = h5py.File(data_file, 'r', locking=False)
    return data_arr

def get_2Dtest_data(config=None):
    import h5py
    
    test_data_path = config.dataset.test_data_file
    test_data = h5py.File(test_data_path)
    u = test_data['u_fluc'][:]
    v = test_data['v_fluc'][:]

    x, y, t  = test_data['x'][:], test_data['y'][:], test_data['t'][:]

    if config.dataset.ds_ratio == 5 :
        u = u[:, :-1, :-1]
        v = v[:, :-1, :-1]
        
        x = x[:-1]
        y = y[:-1]
        nx, ny = u.shape[1:]
        assert (nx, ny) == (300, 100)

    elif config.dataset.ds_ratio == 2 or config.dataset.ds_ratio == 1:
        u = u[:, :-13, :-5]
        v = v[:, :-13, :-5] 
                      
        x = x[:-13]
        y = y[:-5]
        nx, ny = u.shape[1:]
        assert (nx, ny) == (288, 96)
        # shape = (288,96) -> (144,48) # only use upto 4 layers

    u = u[:, ::config.dataset.ds_ratio, ::config.dataset.ds_ratio]
    v = v[:, ::config.dataset.ds_ratio, ::config.dataset.ds_ratio]
    x = x[::config.dataset.ds_ratio]
    y = y[::config.dataset.ds_ratio]

    assert u.shape[1:] == v.shape[1:] == config.model.image_size
    
    u = np.expand_dims(u, axis=1)  
    v = np.expand_dims(v, axis=1)  
    
    # Concatenate along the new axis
    test_data = np.concatenate((u, v), axis=1)
    assert test_data.shape[2:] == u.shape[2:] ==config.model.image_size,  f"Shape mismatch: test_data.shape[2:] = {test_data.shape[2:]}, u.shape[2:] = {u.shape[2:]}"
    print(f"Shape of test data is {test_data.shape}")
    
    return test_data, x, y, t

def remove_indices_from_array(array=None, indices=None, axis=0):
    return np.delete(array, indices, axis=axis)

class OneObs2D(Dataset):
    
    def __init__(self, data_file=None, filetype="hdf5", transform=None, ds_ratio=1, normalize=True, image_size=None):
        
        assert data_file is not None
        self.normalize = normalize
        self.transform = transform

        if filetype == "hdf5":
            
            #------------------------------------------------------------------
            data_arr = read_hdf5(data_file)
            u = np.asarray(data_arr["u_fluc"][:], dtype = np.float32)   # time, nx, ny
            v = np.asarray(data_arr["v_fluc"][:], dtype = np.float32)   # time, nx, ny
            x = np.asarray(data_arr["x"][:])
            y = np.asarray(data_arr["y"][:])
            t = np.asarray(data_arr["t"][:])
            means = np.asarray(data_arr["means"][:], dtype = np.float32)
            
            nx, ny = 301, 101
            assert u.shape[1:] == (nx, ny)  and  v.shape[1:] == (nx, ny)
            assert x.shape == (nx,) and y.shape == (ny,) and t.shape == (u.shape[0], 1 ) and t.shape== (v.shape[0], 1) 
            assert means.shape[1:] == (nx, ny)
            #------------------------------------------------------------------

            if ds_ratio == 5 :
                u = u[:, :-1, :-1]
                v = v[:, :-1, :-1]
                means = means[:, :-1, :-1]
                x = x[:-1]
                y = y[:-1]
                nx, ny = u.shape[1:]
                assert (nx, ny) == (300, 100)
                # shape = (300,100) -> (60,20) # only use 2 layers
                
            elif ds_ratio == 2 or ds_ratio == 1:
                u = u[:, :-13, :-5]
                v = v[:, :-13, :-5] 
                means = means[:, :-13, :-5]              
                x = x[:-13]
                y = y[:-5]
                nx, ny = u.shape[1:]
                assert (nx, ny) == (288, 96)
                # shape = (288,96) -> (144,48) # only use upto 4 layers

            else:
                print(f"ds_ratio {ds_ratio} not supported")
                raise NotImplementedError
            
            #------------------------------------------------------------------
            
            #downsampling by ds_ratio
            u = u[:, ::ds_ratio, ::ds_ratio]
            v = v[:, ::ds_ratio, ::ds_ratio]

            assert u.shape[1:] == v.shape[1:] == image_size

            self.x = x[::ds_ratio]
            self.y = y[::ds_ratio]
            self.t = t
            self.means = means[:, ::ds_ratio, ::ds_ratio]
            self.data = np.stack((u,v), axis = 1 )   # time, nc, nx, ny

            self.u_min, self.u_max = np.min(u), np.max(u)
            self.v_min, self.v_max = np.min(v), np.max(v)

            #------------------------------------------------------------------------------
            assert self.data.shape[1:] == (2, nx//ds_ratio, ny//ds_ratio) and self.data.dtype == np.float32
            
        else:

            raise NotImplementedError
        
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.normalize:
            image = self.__normalize(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __normalize(self, x):
        
        # x shape = (2, h, w)
        eps = 1e-9
        center = np.array([self.u_min, self.v_min]).reshape((2,1,1))
        scale = np.array([self.u_max - self.u_min, self.v_max - self.v_min]).reshape((2,1,1))
        x_scaled =  (x - center) / (scale + eps)
        return ( 2 * x_scaled ) - 1
            

        

class OneObs3D(Dataset):
    
    def __init__(self, data_root=None, data_file=None, filetype="hdf5", transform=None, ds_ratio=1):
        
        assert data_file is not None
        self.data_root = data_root
        self.data_file = data_file
        self.transform = transform
        self.ds_ratio = ds_ratio

        if filetype == "hdf5":
            
            data_arr = read_hdf5(osp.join(self.data_root, self.data_file))
            print(f"{data_file} opened")

            self.data = (data_arr['u_fluc'], data_arr['v_fluc'], data_arr['w_fluc'])
            assert len(self.data[0]) == len(self.data[1]) and  len(self.data[1]) == len(self.data[2])
            
            self.data_len = len(self.data[0])

        else:
            raise NotImplementedError
        
            
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        sample = torch.stack([torch.as_tensor(self.data[j][idx], dtype=torch.float32) for j in range(len(self.data))], dim=0)
        # shape = (num_channels, dim1, dim2, dim3, ..,)
        assert sample.shape == (3,144,63,26)

        if torch.any(torch.isnan(sample)):
           sample[torch.isnan(sample)] = 0
           print(f"NaNs found in data samples at idx={idx} in {self.data_file}, replaced NaNs with 0")     

        if self.ds_ratio == 1:
            padded_sample = torch.zeros((3,144,64,32), dtype=torch.float32)  #TODO: efficient way to do this using torch.nn.functional.pad
            padded_sample[:,:,:63,:26] = sample
            sample = padded_sample
            assert sample.shape == (3,144,64,32)

        elif self.ds_ratio == 2:
            sample = sample[:, ::2, ::2, ::2]
            #print(f"sample shape is {sample.shape}")
            assert sample.shape == (3, 72, 32, 13)
            padded_sample = torch.zeros((3,72,32,16), dtype=torch.float32)  #TODO: efficient way to do this using torch.nn.functional.pad
            padded_sample[:, :, :, :13] = sample
            sample = padded_sample
            assert sample.shape == (3,72,32,16)

        else:
            raise NotImplementedError

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def get_transform(transform):
    if transform is None:
        return None
    else:
        raise NotImplementedError


def get_dataset(config):

    if config.dataset.name.lower() == "oneobs2d":
        
        return OneObs2D(data_file = config.dataset.data_file, filetype = config.dataset.filetype.lower(), 
                        transform = get_transform(config.dataset.transform), ds_ratio = config.dataset.ds_ratio,
                        normalize = config.dataset.normalize, image_size = config.model.image_size)

    elif config.dataset.name.lower() == "oneobs3d":
        
        if isinstance(config.dataset.data_file, list):

            dsets = [OneObs3D(data_root=config.dataset.data_root, data_file=dset, 
                            filetype = config.dataset.filetype.lower(), transform = get_transform(config.dataset.transform), ds_ratio = config.dataset.ds_ratio) for dset in config.dataset.data_file]
                
            return torch.utils.data.Subset(torch.utils.data.ConcatDataset(dsets), indices=torch.arange(50000))

        
        elif isinstance(config.dataset.data_file, str):
            
            return OneObs3D(data_root=config.dataset.data_root, data_file=config.dataset.data_file, 
                            filetype = config.dataset.filetype.lower(), transform = get_transform(config.dataset.transform), ds_ratio = config.dataset.ds_ratio)   
        
        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError