import mcubes
import numpy as np

vertices, triangles = mcubes.marching_cubes(np.load("test_0.npy"), 0)
mcubes.export_obj(vertices, triangles, 'test_0.obj')
print('ok')
exit()