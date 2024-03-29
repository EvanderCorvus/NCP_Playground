import numpy as np
import torch as tr
def force(x,y,U0,type='mexican'):
    if type == 'mexican':
        r = np.sqrt(x**2+y**2)
        bool = r>0.5
        fr = -64*U0*(r**2-0.25)
        fr[bool] = 0
        F_x = fr*x
        F_y = fr*y
        return F_x,F_y

# ToDo: Move everything to GPU
class Box():
    def __init__(self, width, height, center=None, device = 'cuda'):
        self.width = width
        self.height = height
        if center is None:
            self.center = tr.zeros(2).to(device)
            self.centerX = self.center[0]
            self.centerY = self.center[1]
        else:
            self.center = center.to(device)
            self.centerX = center[:,0]
            self.centerY = center[:,1]
    
    def contains(self, x, y):
        bool_tensor = tr.logical_and(tr.logical_and(x > self.centerX - self.width / 2, x < self.centerX + self.width / 2),
                                        tr.logical_and(y > self.centerY - self.height / 2, y < self.centerY + self.height / 2))
        return bool_tensor
    
    def sample(self):
        x = tr.random.uniform(self.centerX-self.width/2, self.centerX+self.width/2)
        y = tr.random.uniform(self.centerY-self.height/2, self.centerY+self.height/2)
        # raise Exception(x.shape, y.shape)
        return tr.tensor([x,y])

class Circle2D():
    def __init__(self, radius, center=None, device = 'cuda'):
        self.radius = radius
        if center is None:
            self.center = tr.zeros(2).to(device)
        else:
            self.center = center.to(device)

    def contains(self, x, y):
        bool = tr.sqrt((x-self.center[0])**2 + (y-self.center[1])**2) < self.radius
        return bool
    
    def sample(self):
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, self.radius)
        x = self.center[:,0] + r*np.cos(theta)
        y = self.center[:,1] + r*np.sin(theta)
        return np.array([x,y])