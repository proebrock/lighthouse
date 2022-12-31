import numpy as np
import open3d as o3d


class Rays:

    def __init__(self, origs, dirs):
        self.origs = np.reshape(np.asarray(origs), (-1, 3))
        self.dirs = np.reshape(np.asarray(dirs), (-1, 3))
        # Make sure origs and dirs have same size
        if self.origs.shape[0] == self.dirs.shape[0]:
            pass
        elif (self.origs.shape[0] == 1) and (self.dirs.shape[0] > 1):
            n = self.dirs.shape[0]
            self.origs = np.tile(self.origs, (n, 1))
        elif (self.origs.shape[0] > 1) and (self.dirs.shape[0] == 1):
            n = self.origs.shape[0]
            self.dirs = np.tile(self.dirs, (n, 1))
        else:
            raise ValueError(f'Invalid values for ray origins (shape {self.origs.shape}) and ray directions (shape {self.dirs.shape})')



    def __len__(self):
        return self.origs.shape[0]



    def filter(self, mask):
        return Rays(self.origs[mask, :], self.dirs[mask, :])



    def points(self, scales):
        return self.origs + scales[:, np.newaxis] * self.dirs



    def normalize(self):
        dirslen = np.sqrt(np.sum(np.square(self.dirs), axis=1))
        nz_mask = ~np.isclose(dirslen, 0.0)
        self.dirs[nz_mask] /= dirslen[nz_mask, np.newaxis]



    def to_tensor_rays(self):
        return o3d.core.Tensor(np.hstack(( \
            self.origs.astype(np.float32), \
            self.dirs.astype(np.float32))))
