from numpy.random import seed
seed(9999)
from tensorflow import set_random_seed
set_random_seed(9999)

import tensorflow as tf
import numpy as np
from model_resnet import gen_atom, dis
from configparser import SafeConfigParser
import os
os.environ['PYTHONHASHSEED'] = '0'
import sys
import math
import pickle
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
from tf_energy import *
from tensorflow.python import debug as tf_debug


energy_masks = np.load("./train_data/energy_masks.npy")
energy_masks = np.transpose(energy_masks, (1,0))

curr_masks = np.load("./train_data/curr_masks.npy")
prev_masks = np.load("./train_data/prev_masks.npy")
curr_seq = np.load("./train_data/curr_seqs.npy")
prev_seq = np.load("./train_data/prev_seqs.npy")
masses = np.load("./train_data/masses.npy")
mask_ndx = np.load("./train_data/masks_ndx.npy")
curr_atoms = np.load("./train_data/curr_atoms.npy")
prev_atoms = np.load("./train_data/prev_atoms.npy")
environment_atoms = np.load("./train_data/aa_env.npy")
bb_beads = np.load("./train_data/bb_beads.npy")
sc_beads = np.load("./train_data/sc_beads.npy")

#Train/Val Split
#n_train = 6118
N_data_tot, _, _ = curr_atoms.shape
ndx_list = list(range(0, N_data_tot))
np.random.shuffle(ndx_list)
ndx_train = ndx_list[:int(0.9*N_data_tot)]
ndx_val = ndx_list[int(0.9*N_data_tot):]

mask_ndx, mask_ndx_val = mask_ndx[ndx_train], mask_ndx[ndx_val]
bb_beads, bb_beads_val = bb_beads[ndx_train], bb_beads[ndx_val]
sc_beads, sc_beads_val = sc_beads[ndx_train], sc_beads[ndx_val]
curr_atoms, curr_atoms_val = curr_atoms[ndx_train], curr_atoms[ndx_val]
prev_atoms, prev_atoms_val = prev_atoms[ndx_train], prev_atoms[ndx_val]
environment_atoms, environment_atoms_val = environment_atoms[ndx_train], environment_atoms[ndx_val]

_, max_curr_seq_len = curr_seq.shape
_, max_prev_seq_len = prev_seq.shape

N_data, seq_len, dim = curr_atoms.shape
N_data_val, _, _ = curr_atoms_val.shape

N_masks, max_seq_len,  N_interactions, max_atoms_ia = curr_masks.shape

N_atomtypes = 4

cfg = SafeConfigParser()
cfg.read('config.ini')

shift = tf.range(1, 8, 1)

sigma = float( cfg.getint('grid', 'sigma'))
ds = float( cfg.getint('grid', 'length') ) / float( cfg.getint('grid', 'max_resolution') )

#Map coords to grid
def map_to_grid(coord_inp, grid_size, ds, sigma):
    grid = tf.range(- int(grid_size/2),int(grid_size/2), dtype= tf.float32)
    grid = tf.add(grid, 0.5)
    grid = tf.scalar_mul(ds, grid)

    X,Y,Z = tf.meshgrid(grid, grid, grid, indexing='ij')
    grid = tf.stack([X,Y,Z])
    grid = tf.expand_dims(grid, 0)
    grid = tf.expand_dims(grid, 0)
    
    coords = tf.expand_dims(coord_inp, dim =-1)
    coords = tf.expand_dims(coords, dim =-1)
    coords = tf.expand_dims(coords, dim =-1)
    
    grid = tf.subtract(grid, coords)
    grid = tf.square(grid)
    grid = tf.reduce_sum(grid, axis = 2)
    
    #cos = tf.cos(tf.sqrt(grid) *2*math.pi/sigma)
    #grid = tf.sqrt(grid)
    
    grid = tf.divide(grid, sigma)
    grid = tf.scalar_mul(-1.0, grid)    
    
    grid = tf.exp(grid)

    grid = tf.transpose(grid, [0,2,3,4,1])
    
    return grid


def map_env_to_grid(coord_inp, grid_size, ds, sigma):
    grid = tf.range(- int(grid_size/2),int(grid_size/2), dtype= tf.float32)
    grid = tf.add(grid, 0.5)
    grid = tf.scalar_mul(ds, grid)

    X,Y,Z = tf.meshgrid(grid, grid, grid, indexing='ij')
    grid = tf.stack([X,Y,Z])
    grid = tf.expand_dims(grid, 0)
    grid = tf.expand_dims(grid, 0)
    grid = tf.expand_dims(grid, 0)
    
    coords = tf.expand_dims(coord_inp, dim =-1)
    coords = tf.expand_dims(coords, dim =-1)
    coords = tf.expand_dims(coords, dim =-1)
    
    grid = tf.subtract(grid, coords)
    grid = tf.square(grid)
    grid = tf.reduce_sum(grid, axis = 3)

    #cos = tf.cos(tf.sqrt(grid) *1.0)

    
    grid = tf.divide(grid, sigma)
    grid = tf.scalar_mul(-1.0, grid)  
    
    grid = tf.exp(grid)

    grid = tf.reduce_sum(grid, axis = 2)
    #grid = tf.expand_dims(grid, dim =-1)
    grid = tf.transpose(grid, [0,2,3,4,1])
  
    return grid

def average_blob_pos(grid):
    grid_size =  cfg.getint('grid', 'max_resolution')
    g = tf.range(- int(grid_size/2),int(grid_size/2), dtype= tf.float32)
    g = tf.add(g, 0.5)
    
    #g = tf.constant([-1.0,1.0], dtype= tf.float32)
    
    g = tf.scalar_mul(ds, g)
    X,Y,Z = tf.meshgrid(g, g, g, indexing='ij')
    
    X = tf.expand_dims(X, 0)
    X = tf.expand_dims(X, -1)
    
    Y = tf.expand_dims(Y, 0)
    Y = tf.expand_dims(Y, -1)
    
    Z = tf.expand_dims(Z, 0)
    Z = tf.expand_dims(Z, -1)    
    
    grid_sum = tf.reduce_sum(grid, axis = [1,2,3], keep_dims =  True)
    
    
    #achtung achtung. zeile ist nur drinne weil sonst nans produziert werden wenn energy loss eingeschaltet wird... wahrscheinlich weil trotz tf.gather 
    #alle gradienten berechnet werden und dann nur mit 0 oder 1 maskiert werden... 
    #grid_sum = tf.clip_by_value(grid_sum, 0.0000000000000000001, 100.0)   
    grid_sum = tf.add(grid_sum, 1E-20)
    
    grid = tf.divide(grid, grid_sum)
    
    X = tf.multiply(grid, X)
    X = tf.reduce_sum(X, axis = [1,2,3])
    
    Y = tf.multiply(grid, Y)
    Y = tf.reduce_sum(Y, axis = [1,2,3])

    Z = tf.multiply(grid, Z)
    Z = tf.reduce_sum(Z, axis = [1,2,3])    
    
    Coords = tf.stack([X,Y,Z], axis = 2)
    
    
    #Coords = tf.where(tf.is_nan(Coords), tf.zeros_like(Coords), Coords)
    return Coords

def center_of_mass(coords):
    carbons = tf.gather(coords, ndx_c_b1, axis =1)
    hydrogens = tf.gather(coords, ndx_h_b1, axis =1)

    com_c = tf.reduce_mean(carbons, axis = 1)
    com_h = tf.reduce_mean(hydrogens, axis = 1)
    
    com = tf.divide(tf.add(tf.multiply(com_c, 12.0),com_h), 13.0)
    
    #com = tf.subtract(com, -2.0)
    
    return com




def is_C(s):
    gate = tf.subtract(1,s)
    gate = tf.abs(gate)
    gate = tf.sign(gate)
    gate = tf.subtract(1,gate)
    return tf.cast(gate, tf.float32)

def is_H(s):
    gate = tf.subtract(2,s)
    gate = tf.abs(gate)
    gate = tf.sign(gate)
    gate = tf.subtract(2,gate)
    return tf.cast(gate, tf.float32)


def new_atom(atoms, x):
    curr_mask,prev_mask, z, shift, type_input = x
    #curr_mask = mask[0]
    #rev_mask = mask[1]
    
    curr_atoms, prev_atoms, env_atoms, cg_beads, dis_input, is_training = atoms

    gen_input = tf.add(get_env(curr_atoms, curr_mask) , get_env(prev_atoms, prev_mask))
    
    #aa_env = tf.add( tf.reduce_sum(curr_atoms, axis = 4, keep_dims= True) , tf.reduce_sum(prev_atoms, axis = 4, keep_dims= True) )  
    gen_input = tf.concat([gen_input, env_atoms, cg_beads], axis = 4)
    
    new_atom = gen_atom(z,gen_input, type_input, is_training)      
    
    new_atom = tf.multiply(new_atom, type_input)
    
    dis_input = tf.concat([gen_input, new_atom], axis = 4)

    curr_atoms = tf.roll(curr_atoms, -shift +1, axis=4)       
    curr_atoms = tf.concat([curr_atoms[:,:,:,:,1:,:], tf.expand_dims(new_atom, axis = 4)], 4)
    curr_atoms = tf.roll(curr_atoms, shift, axis=4)
    
    
    return (curr_atoms, prev_atoms, env_atoms, cg_beads, dis_input, is_training)

def target_atom(atoms, x):
    curr_mask,prev_mask, shift, atom = x
    #curr_mask = mask[0]
    #prev_mask = mask[1]
    
    curr_atoms, prev_atoms, env_atoms, cg_beads, dis_input = atoms

    gen_input = tf.add(get_env(curr_atoms, curr_mask) , get_env(prev_atoms, prev_mask))

    gen_input = tf.concat([gen_input, env_atoms, cg_beads], axis = 4)
    
    dis_input = tf.concat([gen_input, atom], axis = 4)

    curr_atoms = tf.roll(curr_atoms, -shift +1, axis=4)    
    curr_atoms = tf.concat([curr_atoms[:,:,:,:,1:,:], tf.expand_dims(atom, axis = 4)], 4)
    curr_atoms = tf.roll(curr_atoms, shift, axis=4)    
  
    return (curr_atoms, prev_atoms, env_atoms, cg_beads, dis_input)



def get_env(atoms, mask):
    grids = tf.map_fn(grab_atoms, (atoms, mask), dtype=tf.float32)
    grids = tf.reduce_sum(grids, [5])     
    grids = tf.reshape(grids, [cfg.getint('model', 'batchsize'),8,8,8, N_interactions * N_atomtypes])
    return grids    
    
def get_d_loss(t):
    real, fake, dis_mask, is_training= t
    
    D_real = tf.multiply( dis(real, is_training) , dis_mask)    
    D_fake = tf.multiply( dis(fake, is_training) , dis_mask)
    
    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
    
    
    # gradient penalty
    differences2 = fake - real
    alpha_gp2 = tf.random_uniform(shape=[cfg.getint('model', 'batchsize'), 1, 1, 1, 1], minval=0., maxval=1.)
    interpolates2 = real + (alpha_gp2 * differences2)
    D_int2= dis(interpolates2, is_training)
    
    gradients2 = tf.gradients(D_int2, [interpolates2])[0]
    slopes2 = tf.sqrt(tf.reduce_sum(tf.square(gradients2), reduction_indices=[1, 2, 3, 4]))
    
    slopes2 = (slopes2 - 1.) ** 2 
    slopes2 = tf.multiply( slopes2, dis_mask)
    gradient_penalty2 = tf.reduce_mean(  slopes2 )

    #gradient_penalty2 = tf.reduce_mean(   (slopes2 - 1.) ** 2   )
    D_loss += 10 * gradient_penalty2
    
    #Drift loss
    D_loss += 0.001 * tf.reduce_mean(tf.square(D_real - 0.0)) 
    
    return D_loss

def get_g_loss(t):
    fake, dis_mask, is_training = t    
    D_fake = tf.multiply( dis(fake, is_training) , dis_mask)
    #D_fake = dis_part2(fake)
    G_loss = -tf.reduce_mean(D_fake)
    return G_loss

def get_target_atoms(atoms_b2, new_atom):
    target_atoms = tf.concat([new_atom, atoms_b2[:,:,:,:,0:3]], 4)
    return target_atoms  

def grab_atoms(x):
    atoms , indices = x
    print(atoms)
    print(indices)
    return tf.gather(atoms, indices, axis = 3)

def get_bond_energy(x):
    atoms, indices = x
    return tf_energy_bond(atoms, indices)

def get_angle_energy(x):
    atoms, indices = x
    return tf_energy_angle(atoms, indices)

def get_pdih_energy(x):
    atoms, indices = x
    return tf_energy_prop_dihedrals(atoms, indices)

def get_idih_energy(x):
    atoms, indices = x
    return tf_energy_iprop_dihedrals(atoms, indices)

def get_lj_intra_energy(x):
    atoms, indices = x
    return tf_energy_lj_intramol(atoms, indices)

def get_lj_env_energy(x):
    atoms_mol, atoms_env, seq = x
    return tf_energy_lj_env(atoms_mol, atoms_env, seq)

def make_atomtype_channels(x):
    atoms, seq = x
    channels = []
    for chn in range(1,5):
        type_mask = tf.cast(tf.equal(seq, 1), tf.float32)
        type_mask = tf.expand_dims(type_mask, 0)
        type_mask = tf.expand_dims(type_mask, 0)
        type_mask = tf.expand_dims(type_mask, 0)
        atomtype_channel = tf.multiply(atoms, type_mask)
        atomtype_channel = tf.reduce_sum(atomtype_channel, axis = 3, keepdims = True)
        channels.append(tf.identity(atomtype_channel))
    return tf.concat(channels, axis=3)

def com(coords, masses):
    #coords (32,12,3)
    #masses (32,12)
    
    tot_mass = tf.reduce_sum(masses, axis=1, keep_dims = True)
    com = tf.multiply(coords, tf.expand_dims(masses, axis=-1))
    com = tf.reduce_sum(com, axis= 1)
    com = tf.divide(com, tot_mass)
    
    return com
    


#Placeholders
learning_rate = tf.placeholder(tf.float32)
energy_mol_learning_rate = tf.placeholder(tf.float32)
energy_env_learning_rate = tf.placeholder(tf.float32)
#blindness_mol = tf.placeholder(tf.float32, shape=())
#blindness_env = tf.placeholder(tf.float32, shape=())

global_step = tf.Variable(0, name='global_step', trainable=False)
is_training = tf.placeholder_with_default(False, (), 'is_training')

#noise
z = tf.placeholder(tf.float32, [7, cfg.getint('model', 'batchsize'), int(cfg.getint('model', 'noise_dim') )])

curr_atoms_input = tf.placeholder(tf.float32, shape=(cfg.getint('model', 'batchsize'), max_curr_seq_len,3), name='current_atoms')
prev_atoms_input = tf.placeholder(tf.float32, shape=(cfg.getint('model', 'batchsize'), max_prev_seq_len,3), name='prev_atoms')

env_atoms_input = tf.placeholder(tf.float32, shape=(cfg.getint('model', 'batchsize'), 4, 20, 3), name='env_atoms')


bb_beads_input = tf.placeholder(tf.float32, shape=(cfg.getint('model', 'batchsize'), None,3), name='bb_beads_input')
sc_beads_input = tf.placeholder(tf.float32, shape=(cfg.getint('model', 'batchsize'), None,3), name='sc_beads_input')


curr_seq_input = tf.placeholder(tf.int32, shape=(cfg.getint('model', 'batchsize'), max_curr_seq_len), name='curr_seq_input')
prev_seq_input = tf.placeholder(tf.int32, shape=(cfg.getint('model', 'batchsize'), max_prev_seq_len), name='prev_seq_input')

curr_masks_input = tf.placeholder(tf.int32, shape=(max_curr_seq_len,cfg.getint('model', 'batchsize'), N_interactions, None), name='curr_mask') 
prev_masks_input = tf.placeholder(tf.int32, shape=(max_curr_seq_len,cfg.getint('model', 'batchsize'), N_interactions, None), name='prev_mask') 

#sequences_input = tf.placeholder(tf.int32, shape=(cfg.getint('model', 'batchsize'),2, 12), name='sequences_input')
#masks_input = tf.placeholder(tf.int32, shape=(max_seq_len,2,cfg.getint('model', 'batchsize'), 4, 5), name='prev_atoms') 
masses_input = tf.placeholder(tf.float32, shape=(cfg.getint('model', 'batchsize'), max_curr_seq_len), name='masses_input')


bond_indices = tf.placeholder(tf.int32, shape=(cfg.getint('model', 'batchsize'),3, None), name='bond_indices') 
angle_indices = tf.placeholder(tf.int32, shape=(cfg.getint('model', 'batchsize'),4, None), name='angle_indices') 
pdih_indices = tf.placeholder(tf.int32, shape=(cfg.getint('model', 'batchsize'),5, None), name='pdih_indices') 
idih_indices = tf.placeholder(tf.int32, shape=(cfg.getint('model', 'batchsize'),5, None), name='idih_indices') 
lj_indices = tf.placeholder(tf.int32, shape=(cfg.getint('model', 'batchsize'),3, None), name='lj_indices') 

#encode atomtype
curr_atype_mask = tf.one_hot(curr_seq_input-1, 4)
curr_atype_mask = tf.cast(curr_atype_mask, tf.float32) #(BS,12,4)
#curr_atype_mask = tf.transpose(curr_atype_mask, [1,0])
curr_atype_mask = tf.expand_dims(curr_atype_mask,1)
curr_atype_mask = tf.expand_dims(curr_atype_mask,1)
curr_atype_mask = tf.expand_dims(curr_atype_mask,1)#(BS,1,1,1,12,4)
type_input = tf.transpose(curr_atype_mask, [4,0,1,2,3,5]) #(12,Bs,1,1,1,4)

prev_atype_mask = tf.one_hot(prev_seq_input-1, 4)
prev_atype_mask = tf.cast(prev_atype_mask, tf.float32) #(BS,12,4)
#prev_atype_mask = tf.transpose(prev_atype_mask, [1,0,2])
prev_atype_mask = tf.expand_dims(prev_atype_mask,1)
prev_atype_mask = tf.expand_dims(prev_atype_mask,1)
prev_atype_mask = tf.expand_dims(prev_atype_mask,1)


#Mask for discriminator output
dis_mask = curr_seq_input
dis_mask = tf.clip_by_value(dis_mask, 0, 1)
dis_mask = tf.transpose(dis_mask, [1,0])
dis_mask = tf.cast(dis_mask, tf.float32)#(12,BS)


#Mask for atoms
atom_mask = curr_seq_input
atom_mask = tf.clip_by_value(atom_mask, 0, 1)
atom_mask = tf.cast(atom_mask, tf.float32)
atom_mask = tf.expand_dims(atom_mask, 1)
atom_mask = tf.expand_dims(atom_mask, 1)
atom_mask = tf.expand_dims(atom_mask, 1)#(BS,1,1,1,12)

#make input
#initial_curr_atoms = tf.zeros([cfg.getint('model', 'batchsize'),8,8,8,12,4])
initial_dis_atoms = tf.zeros([cfg.getint('model', 'batchsize'),8,8,8,4*4+10])
c_atoms = map_to_grid(curr_atoms_input, cfg.getint('grid', 'max_resolution'),ds, sigma) #(BS,8,8,8,12)
c_atoms = tf.multiply(tf.expand_dims(c_atoms,-1), curr_atype_mask) #(BS,8,8,8,12,4)

c_atoms = tf.add(c_atoms, 10E-20) #prevent zero input for discriminator, and average_blob_pos

c_atoms_trans = tf.transpose(c_atoms, [4,0,1,2,3,5]) #(12,BS,8,8,8,4)

p_atoms = map_to_grid(prev_atoms_input, cfg.getint('grid', 'max_resolution'),ds, sigma)
p_atoms = tf.multiply(tf.expand_dims(p_atoms,-1), prev_atype_mask) #(BS,8,8,8,12,4)

p_atoms = tf.add(p_atoms, 10E-20) #prevent zero input for discriminator, and average_blob_pos


cg_beads = tf.concat([tf.expand_dims(bb_beads_input, axis=1), tf.expand_dims(sc_beads_input, axis=1)], axis = 1)
cg_beads = map_env_to_grid(cg_beads, cfg.getint('grid', 'max_resolution'),ds)

env_atoms = map_env_to_grid(env_atoms_input, cfg.getint('grid', 'max_resolution'),ds, sigma)

real_curr_atom_seq , prev_atoms_seq ,_ ,_ , real_dis_input = tf.scan(target_atom, (curr_masks_input, prev_masks_input,shift, c_atoms_trans), (c_atoms, p_atoms, env_atoms, cg_beads, initial_dis_atoms))
fake_curr_atom_seq,_, _, _, fake_dis_input, _= tf.scan(new_atom, (curr_masks_input, prev_masks_input,z,shift, type_input), (c_atoms, p_atoms, env_atoms, cg_beads, initial_dis_atoms, is_training))

target_atoms = tf.concat([p_atoms, c_atoms], axis = 4)
target_atoms = tf.reduce_sum(target_atoms, axis = 5)

#fake_curr_atoms = tf.multiply(fake_curr_atom_seq[11], curr_atype_mask)
fake_atoms = tf.concat([p_atoms, fake_curr_atom_seq[6]], axis = 4)
fake_atoms = tf.reduce_sum(fake_atoms, axis = 5)


is_training_stacked = tf.stack([is_training]*max_curr_seq_len)

d_losses = tf.map_fn(get_d_loss, (real_dis_input,fake_dis_input,  dis_mask, is_training_stacked), dtype = tf.float32)
#d_losses = tf.Print(d_losses,[d_losses], "D_loss")

g_losses = tf.map_fn(get_g_loss, (fake_dis_input, dis_mask, is_training_stacked), dtype = tf.float32)


D_tot_loss = tf.reduce_sum(d_losses)
G_loss_adv = tf.reduce_sum(g_losses)


G_tot_loss = G_loss_adv

#Type Loss

fake_type = tf.reduce_sum(fake_curr_atom_seq[6], axis=[1,2,3]) #(32,8,8,8,12,4) ->(BS,12,4)
target_type = tf.reduce_sum(c_atoms, axis=[1,2,3]) #(BS,1,1,1,12,4) -> (BS,12,4)
G_loss_type = tf.subtract(target_type, fake_type)
G_loss_type = tf.square(G_loss_type)
#mask dummie atoms
G_loss_type = tf.reduce_sum(G_loss_type, axis = 2)
G_loss_type = tf.multiply(G_loss_type, tf.transpose(dis_mask, [1,0]))
G_loss_type = tf.reduce_sum(G_loss_type)


#G_tot_loss = tf.add(G_tot_loss,G_loss_type*0.001)


fake_coords = average_blob_pos(fake_atoms)
#target_coords = average_blob_pos(target_atoms)
target_coords = tf.concat([prev_atoms_input, curr_atoms_input], axis = 1)

#Center of Mass
real_com = com(target_coords[:,max_prev_seq_len:,:], masses_input)
fake_com = com(fake_coords[:,max_prev_seq_len:,:], masses_input)
G_loss_com = tf.subtract(real_com, fake_com)
G_loss_com = tf.square(G_loss_com)
G_loss_com = tf.reduce_sum(G_loss_com)

G_tot_loss = tf.add(G_tot_loss,G_loss_com)


print("-------")
print(target_coords)
print("-------")

real_bond_energy = tf.map_fn(get_bond_energy, (target_coords, bond_indices), dtype=tf.float32)
fake_bond_energy = tf.map_fn(get_bond_energy, (fake_coords, bond_indices), dtype=tf.float32)
Bond_loss = tf.reduce_mean(tf.abs(tf.subtract(real_bond_energy, fake_bond_energy)))
Bond_loss_scaled = tf.multiply(Bond_loss , energy_mol_learning_rate* 10)

real_angle_energy = tf.map_fn(get_angle_energy, (target_coords, angle_indices), dtype=tf.float32)
fake_angle_energy = tf.map_fn(get_angle_energy, (fake_coords, angle_indices), dtype=tf.float32)
Angle_loss = tf.reduce_mean(tf.abs(tf.subtract(real_angle_energy, fake_angle_energy)))
Angle_loss_scaled = tf.multiply(Angle_loss , energy_mol_learning_rate)

real_pdih_energy = tf.map_fn(get_pdih_energy, (target_coords, pdih_indices), dtype=tf.float32)
fake_pdih_energy = tf.map_fn(get_pdih_energy, (fake_coords, pdih_indices), dtype=tf.float32)
Pdih_loss = tf.reduce_mean(tf.abs(tf.subtract(real_pdih_energy, fake_pdih_energy))) 
Pdih_loss_scaled = tf.multiply(Pdih_loss , energy_mol_learning_rate)

real_idih_energy = tf.map_fn(get_idih_energy, (target_coords, idih_indices), dtype=tf.float32)
fake_idih_energy = tf.map_fn(get_idih_energy, (fake_coords, idih_indices), dtype=tf.float32)
Idih_loss = tf.reduce_mean(tf.abs(tf.subtract(real_idih_energy, fake_idih_energy))) 
Idih_loss_scaled = tf.multiply(Idih_loss , energy_mol_learning_rate)

real_lj_intra_energy = tf.map_fn(get_lj_intra_energy, (target_coords, lj_indices), dtype=tf.float32)
fake_lj_intra_energy = tf.map_fn(get_lj_intra_energy, (fake_coords, lj_indices), dtype=tf.float32)
Lj_intra_loss = tf.reduce_mean((tf.abs(tf.subtract(real_lj_intra_energy, fake_lj_intra_energy))))

Lj_intra_loss_scaled = tf.multiply(Lj_intra_loss , energy_mol_learning_rate )

print("-------")
print(target_coords)
print("-------")

real_lj_env_energy = tf.map_fn(get_lj_env_energy, (target_coords[:,max_prev_seq_len:,:], env_atoms_input, curr_seq_input), dtype=tf.float32)
fake_lj_env_energy = tf.map_fn(get_lj_env_energy, (fake_coords[:,max_prev_seq_len:,:], env_atoms_input, curr_seq_input), dtype=tf.float32)
Lj_env_loss = tf.reduce_mean((tf.abs(tf.subtract(real_lj_env_energy, fake_lj_env_energy))))

Lj_env_loss_scaled = tf.multiply(Lj_env_loss , energy_mol_learning_rate )

G_loss_e = tf.add_n([Bond_loss_scaled, Angle_loss_scaled, Pdih_loss_scaled, Idih_loss_scaled, Lj_intra_loss_scaled, Lj_env_loss_scaled])

G_tot_loss = tf.add(G_tot_loss,G_loss_e)
#G_tot_loss = tf.add(G_tot_loss,Lj_env_loss_scaled)


#Get trainable variables
t_vars = tf.trainable_variables()



d_vars = [var for var in t_vars if 'discriminator' in var.name]
total_para = 0
for variable in d_vars:
    shape = variable.get_shape()
    print (variable.name, shape)
    variable_para = 1
    for dim in shape:
        variable_para *= dim.value
    total_para += variable_para
print ("The total number of parameters of D", total_para)

#Gen variables
g_vars = [var for var in t_vars if 'generator' in var.name]
total_para = 0
for variable in g_vars:
    shape = variable.get_shape()
    print (variable.name, shape)
    variable_para = 1
    for dim in shape:
        variable_para *= dim.value
    total_para += variable_para
print ("The total number of parameters of G", total_para)

#tensorboard
tf.summary.scalar('G_loss_total', G_tot_loss)
tf.summary.scalar('G_loss_adv', G_loss_adv)
tf.summary.scalar('G_loss_type', G_loss_type)
tf.summary.scalar('G_loss_e', G_loss_e)
tf.summary.scalar('G_loss_com', G_loss_com)

tf.summary.scalar('Real Bond', tf.reduce_mean(real_bond_energy))
tf.summary.scalar('Real Angle', tf.reduce_mean(real_angle_energy))
tf.summary.scalar('Real Pdih', tf.reduce_mean(real_pdih_energy))
tf.summary.scalar('Real Idih', tf.reduce_mean(real_idih_energy))
tf.summary.scalar('Real LJ_intra', tf.reduce_mean(real_lj_intra_energy))
tf.summary.scalar('Real LJ_env', tf.reduce_mean(real_lj_env_energy))

tf.summary.scalar('Fake Bond', tf.reduce_mean(fake_bond_energy))
tf.summary.scalar('Fake Angle', tf.reduce_mean(fake_angle_energy))
tf.summary.scalar('Fake Pdih', tf.reduce_mean(fake_pdih_energy))
tf.summary.scalar('Fake Idih', tf.reduce_mean(fake_idih_energy))
tf.summary.scalar('Fake LJ_intra', tf.reduce_mean(fake_lj_intra_energy))
tf.summary.scalar('Fake LJ_env', tf.reduce_mean(fake_lj_env_energy))

tf.summary.scalar('Bond Loss', Bond_loss)
tf.summary.scalar('Angle Loss', Angle_loss)
tf.summary.scalar('Pdih Loss', Pdih_loss)
tf.summary.scalar('Idih Loss', Idih_loss)
tf.summary.scalar('LJ_intra Loss', Lj_intra_loss)
tf.summary.scalar('LJ_env Loss', Lj_env_loss)

tf.summary.scalar('D_loss', D_tot_loss)

#Train Ops
d_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.0, beta2=0.9)

g_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.0, beta2=0.9)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_train_op = d_solver.minimize(D_tot_loss, var_list=d_vars)
    g_train_op = g_solver.minimize(G_tot_loss, var_list=g_vars, global_step=global_step)



#Make Dirs for saving
MODEL_DIR = './'+"atom_rnn_augtrainset_notype"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


CHECKPOINT_DIR = MODEL_DIR +'/checkpoints'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
CHECKPOINT_PATH = CHECKPOINT_DIR + '/checkpoint'
SAMPLES_DIR     = MODEL_DIR +'/samples'
if not os.path.exists(SAMPLES_DIR):
    os.makedirs(SAMPLES_DIR)
LOGS_DIR     = MODEL_DIR +'/logs'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def make_dict(lr,lr_energy,ndx_epoch, data="train", training=True):
    rot_mat = rotation_matrix(np.array([0.0,0.0,1.0]), np.random.uniform()*math.pi*2.0)  
    
    if data == "train":
        #idx = np.random.choice(N_data, cfg.getint('model', 'batchsize'))
        idx = ndx_epoch[:cfg.getint('model', 'batchsize')]
        ndx_epoch = ndx_epoch[cfg.getint('model', 'batchsize'):]
        masks_ids = mask_ndx[idx]
        

        env_atoms_ = environment_atoms[idx]   
        curr_atoms_ = curr_atoms[idx]  
        prev_atoms_ = prev_atoms[idx] 
        bb_beads_ = bb_beads[idx]        
        sc_beads_ = sc_beads[idx] 
       
        
        curr_masks_ = np.transpose(curr_masks[mask_ndx[idx]] , (1,0,2,3))
        prev_masks_ = np.transpose(prev_masks[mask_ndx[idx]] , (1,0,2,3))
        curr_seq_ = curr_seq[mask_ndx[idx]]
        prev_seq_ = prev_seq[mask_ndx[idx]]
        
        masses_ = masses[mask_ndx[idx]]
        bond_indices_ = np.array(list(energy_masks[0][mask_ndx[idx]]))
        angle_indices_ = np.array(list(energy_masks[1][mask_ndx[idx]]))
        pdih_indices_ = np.array(list(energy_masks[2][mask_ndx[idx]]))
        idih_indices_ = np.array(list(energy_masks[3][mask_ndx[idx]]))
        lj_indices_ = np.array(list(energy_masks[4][mask_ndx[idx]]))
    else:
        idx = np.random.choice(N_data_val, cfg.getint('model', 'batchsize'))
        masks_ids = mask_ndx_val[idx]

        env_atoms_ = environment_atoms_val[idx]   
        curr_atoms_ = curr_atoms_val[idx]   
        prev_atoms_ =prev_atoms_val[idx]
        bb_beads_ = bb_beads_val[idx]         
        sc_beads_ = sc_beads_val[idx]  
        
        curr_masks_ = np.transpose(curr_masks[mask_ndx_val[idx]] , (1,0,2,3))
        prev_masks_ = np.transpose(prev_masks[mask_ndx_val[idx]] , (1,0,2,3))
        curr_seq_ = curr_seq[mask_ndx_val[idx]]
        prev_seq_ = prev_seq[mask_ndx_val[idx]]
        
        masses_ = masses[mask_ndx_val[idx]]
        bond_indices_ = np.array(list(energy_masks[0][mask_ndx_val[idx]]))
        angle_indices_ = np.array(list(energy_masks[1][mask_ndx_val[idx]]))
        pdih_indices_ = np.array(list(energy_masks[2][mask_ndx_val[idx]]))
        idih_indices_ = np.array(list(energy_masks[3][mask_ndx_val[idx]]))
        lj_indices_ = np.array(list(energy_masks[4][mask_ndx_val[idx]]))




    for r in range(0, cfg.getint('model', 'batchsize')):
        curr_atoms_[r] = np.dot(curr_atoms_[r], rot_mat)
        prev_atoms_[r] = np.dot(prev_atoms_[r], rot_mat)
        bb_beads_[r] = np.dot(bb_beads_[r], rot_mat)
        sc_beads_[r] = np.dot(sc_beads_[r], rot_mat)
        for rr in range(0,4):
            env_atoms_[r,rr] = np.dot(env_atoms_[r,rr], rot_mat)

    batch_z = np.random.normal(0, 1, size=(7, cfg.getint('model', 'batchsize'), int(cfg.getint('model', 'noise_dim'))))

    #print(masks_[0,0])
    feed_dict = {z: batch_z,
                 curr_atoms_input: curr_atoms_,
                 prev_atoms_input: prev_atoms_,
                 bb_beads_input: bb_beads_,
                 sc_beads_input: sc_beads_,
                 curr_masks_input: curr_masks_,
                 prev_masks_input: prev_masks_,

                 curr_seq_input: curr_seq_,
                 prev_seq_input: prev_seq_,
                 masses_input: masses_,
                 bond_indices: bond_indices_,
                 angle_indices: angle_indices_,
                 pdih_indices: pdih_indices_,
                 idih_indices: idih_indices_,
                 lj_indices: lj_indices_,
                 env_atoms_input: env_atoms_,
                 learning_rate: lr,
                 energy_mol_learning_rate: lr_energy,
                 is_training: training}

    return feed_dict, masks_ids, ndx_epoch


#Load model
init = tf.global_variables_initializer()
saver = tf.train.Saver()

def load(sess, saver, load_dir):
    print("Loading model...")
    saver.restore(sess, tf.train.latest_checkpoint(load_dir))
    #saver.restore(sess, "/home/mstieffe/atom_rnn/atom_rnn_t568_amorph_l12_g2_ln_carbonsfirst/checkpoint_conly/checkpoint-4000")
    
merge = tf.summary.merge_all()
  
def e_learn_rate(start, end, step, tot_steps):
    return step/tot_steps * end + start
def learn_rate(start, end, step, tot_steps):
    delta_rate = start - end
    if delta_rate == start:
        delta_rate = 0.0
    return start - step/tot_steps * delta_rate

n_epoch = 12
with tf.Session() as sess:
    
    train_writer = tf.summary.FileWriter( LOGS_DIR+'/train ', sess.graph)
    val_writer = tf.summary.FileWriter( LOGS_DIR+'/val ', sess.graph)

    sess.run(init)
    if cfg.getint('training', 'load_model'):
        load(sess, saver, CHECKPOINT_DIR)
        glob_step = sess.run(global_step)
            
    dl_ave = 0.0
    gl_ave = 0.0
    
    for ep in range(0, n_epoch):
        ndx_epoch = np.array(range(0,N_data))
        np.random.shuffle(ndx_epoch)
        ndx_epoch_dis = np.array(range(0,N_data))
        np.random.shuffle(ndx_epoch_dis)
        ndx_epoch_dis = np.array(list(ndx_epoch)*5)
        max_iters = int(N_data/cfg.getint('model', 'batchsize'))
        
        for i in range(1, max_iters + 1):
            e_learn = e_learn_rate(0.01, 0.0, i , max_iters + 1)
            for j in range(0, cfg.getint('training', 'n_critic')):      
                feed_dict, _, ndx_epoch_dis = make_dict(0.0001, e_learn, ndx_epoch_dis)
                _, dl = sess.run([ d_train_op, D_tot_loss],feed_dict=feed_dict)
    
            print(ep, len(ndx_epoch))
            feed_dict, _, ndx_epoch = make_dict(0.00005, e_learn, ndx_epoch)
            _, g_l, glob_step, bond_e, angle_e, pdih_e, idih_e, lj_intra, lj_env, t_loss, g_c = sess.run([g_train_op, G_tot_loss, global_step, Bond_loss, Angle_loss, Pdih_loss, Idih_loss, Lj_intra_loss, Lj_env_loss, G_loss_type, G_loss_com], feed_dict=feed_dict)
            
    
            dl_ave = dl_ave + dl
    
            gl_ave = gl_ave + g_l
            if i%cfg.getint('record', 'disp_iter') == 0:
                #dl_loc_ave = dl_loc_ave/cfg.getint('record', 'disp_iter')
                dl_ave = dl_ave/cfg.getint('record', 'disp_iter')
                gl_ave = gl_ave /cfg.getint('record', 'disp_iter')
                print('step:',glob_step,max_iters,'D loss:',dl_ave, 'G_loss:',gl_ave, t_loss, g_c, "Energy Loss: ", bond_e, angle_e, pdih_e, idih_e, lj_intra, lj_env, "Rates: ", e_learn, 0.0001, 0.00005)
                dl_ave = 0.0
                gl_ave = 0.0     
    
            if i%cfg.getint('record', 'tensorboard_iter') == 0:
                feed_dict_train, _, _ = make_dict(0.0001, 0.01, ndx_epoch, False)
                feed_dict_val, _, _ = make_dict(0.0001, 0.01, ndx_epoch, "val", False)
    
                #summary_train = sess.run(merge,feed_dict=feed_dict)
                         
                summary_train = sess.run(merge,feed_dict=feed_dict_train)            
                summary_val = sess.run(merge,feed_dict=feed_dict_val)            
                
                train_writer.add_summary(summary_train, glob_step)
                val_writer.add_summary(summary_val, glob_step)
    
    
            if i%cfg.getint('record', 'sample_iter') == 0:
                print("Saving samples in {}".format(SAMPLES_DIR))
                feed_dict_val, mask_ids, _ = make_dict(0.0001, 0.01, ndx_epoch, "val", False)
    
                
                t, f, t_grid, f_grid, d_mask, fdi, rdi = sess.run([target_coords, fake_coords, target_atoms, fake_atoms, dis_mask, fake_dis_input, real_dis_input], feed_dict=feed_dict_val)
                np.save(SAMPLES_DIR+"/target_coords.npy", t)
                np.save(SAMPLES_DIR+"/fake_coords.npy", f)
                np.save(SAMPLES_DIR+"/target_grid.npy", t_grid)
                np.save(SAMPLES_DIR+"/fake_grid.npy", f_grid)
                np.save(SAMPLES_DIR+"/d_mask.npy", d_mask)
                np.save(SAMPLES_DIR+"/fake_dis_input.npy", fdi)
                np.save(SAMPLES_DIR+"/real_dis_input.npy", rdi)            
                np.save(SAMPLES_DIR+"/mask_ndx.npy", mask_ids)            
                
                print('Done saving')      
            """
            if i%cfg.getint('record', 'rec_iter') == 0:
                print("Saving model in {}".format(CHECKPOINT_PATH))
                saver.save(sess, CHECKPOINT_PATH, global_step)
                print('Done saving') 
            """
        print("Saving model in {}".format(CHECKPOINT_PATH))
        saver.save(sess, CHECKPOINT_PATH, global_step)
        print('Done saving') 
