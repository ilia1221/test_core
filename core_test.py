from wavefront import load_obj
import numpy as np
import seaborn as sns
from core import Core


core = Core(device='cpu')


obj_path = './test_shapes/Cone.001.obj'
# obj_path = './test_shapes/Cone.002.obj'
# obj_path = './test_shapes/Torus.1732.obj'
# obj_path = './test_shapes/Torus.1733.obj'

shape = load_obj(obj_path).vertices
shape = np.asarray(shape)

res = core.predict(shape)

sns.barplot(res)

q = 1