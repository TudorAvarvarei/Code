### %pylab notebook
from pyquad.ekin import*
from tqdm import tqdm
import importlib
import sys
sys.path.append("C:\\Users\\tavar\\Desktop\\Thesis\\Code")
from quad_SL.quadcopter_animation import animation

def transform_back(dx,dy,dz,vx,vy,vz,phi,theta,psi):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    R = Rz@Ry@Rx
    x_new, y_new, z_new = -R@[dx, dy, dz]
    vx_new, vy_new, vz_new = R@[vx, vy, vz]
    return x_new, y_new, z_new, vx_new, vy_new, vz_new, phi, theta, psi

datafile = "./../../../../quad_SL/datasets/hover_dataset.npz"
a = np.load(datafile)
num = a['dx'].shape[0]
print(num)

t,dx,dy,dz,vx,vy,vz,phi,theta,psi,p,q,r,omega,u,Mx_ext,My_ext,Mz_ext = (a[key][0:1000] for key in 't dx dy dz vx vy vz phi theta psi p q r omega u Mx_ext My_ext Mz_ext'.split(' '))

importlib.reload(animation)

x,y,z,vx,vy,vz,phi,theta,psi = np.vectorize(transform_back)(dx,dy,dz,vx,vy,vz,phi,theta,psi)
# t,x,y,z,vx,vy,vz,phi,theta,psi,u = (a[key] for key in 't x y z vx vy vz phi theta psi u'.split(' '))

animation.animate(t, x, y, z, phi, theta, psi, u, multiple_trajectories=True, simultaneous=False, step=1, waypoints=[np.array([0,0,0])], colors=[(255,0,0)]*100)