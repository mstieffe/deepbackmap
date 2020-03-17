import tensorflow as tf
import numpy as np
import os
import sys
import math
#import MDAnalysis as mda


w = np.load('1re40.npy').item()
params = w["ff"]
params = params[:,1:]
params = tf.constant(params, tf.float32)

n_atomtypes = len(np.unique(w["atoms_type"]))

radiant_conversion = tf.constant( math.pi / 180.0, tf.float32)

def tf_energy_angle(atoms, indices):
    ndx1 = indices[1]
    ndx2 = indices[2]
    ndx3 = indices[3]
    
    param_ndx = indices[0]
    param = tf.gather(params, param_ndx, axis = 0)
   
    #angles in radiant
    a_0 = tf.multiply(param[:,0] , radiant_conversion)
    #force constant is already per radiant squared
    f_c = param[:,1]

    
    pos1 = tf.gather(atoms, ndx1, axis =0)
    pos2 = tf.gather(atoms, ndx2, axis =0)
    pos3 = tf.gather(atoms, ndx3, axis =0)
    
    vec1 = tf.subtract(pos1, pos2)
    vec2 = tf.subtract(pos3, pos2)    

    norm1 = tf.square(vec1)
    norm1 = tf.reduce_sum(norm1, axis = 1)
    norm1 = tf.sqrt(norm1)    

    norm2 = tf.square(vec2)
    norm2 = tf.reduce_sum(norm2, axis = 1)
    norm2 = tf.sqrt(norm2)


    dot = tf.multiply(vec1, vec2)
    dot = tf.reduce_sum(dot, axis = 1)
    
    norm = tf.multiply(norm1, norm2)

    norm = tf.clip_by_value(norm, 10E-8, 1000.0)
    
    a = tf.divide(dot, norm)
    
    a = tf.clip_by_value(a, -0.9999, 0.9999) #prevent nan because of rounding errors

    #tf.acos should return angle in radiant??
    a = tf.acos(a)

    en = tf.subtract(a, a_0)
    en = tf.square(en)
    en = tf.multiply(en, f_c)
    en = tf.divide(en, 2.0)
    en = tf.reduce_sum(en, axis = 0)
    return en

#bond_constr = tf.constant(1000.0, tf.float32)

def tf_energy_bond(atoms, indices):
    ndx1 = indices[1]
    ndx2 = indices[2]

    param_ndx = indices[0]
    param = tf.gather(params, param_ndx, axis = 0)
    
    a_0 = param[:,0]
    f_c = param[:,1]

    pos1 = tf.gather(atoms, ndx1, axis =0)
    pos2 = tf.gather(atoms, ndx2, axis =0)
    
    dis = tf.subtract(pos1, pos2)
    dis = tf.square(dis)
    dis = tf.reduce_sum(dis, axis = 1)
    dis = tf.sqrt(dis)
    #GROMACS USES NM WHILE THE DISTANCES HERE ARE GIVEN IN ANGSTROM
    dis = tf.divide(dis, 10.0)    

    dis = tf.clip_by_value(dis, 10E-8, 1000.0)

    #dis_0 = tf.square(a_0)

    en = tf.subtract(dis, a_0)
    en = tf.square(en)
    
    #dis2 = tf.cast(dis2, dtype = tf.float32)
    en = tf.multiply(en, f_c / 2.0)
    #en = tf.divide(en, 4.0)
    
    en = tf.reduce_sum(en, axis = 0)
    
    
    #en = tf.clip_by_value(en, 0.0, 2.0)
    #e = tf.add(e, en)
    return en

def tf_energy_lj_intramol(atoms, indices):
    atoms = tf.divide(atoms, 10.0)

    
    ndx1 = indices[1]
    ndx2 = indices[2]

    param_ndx = indices[0]
    param = tf.gather(params, param_ndx, axis = 0)
    
    c6 = param[:,0]
    c12 = param[:,1]

    pos1 = tf.gather(atoms, ndx1, axis =0)
    pos2 = tf.gather(atoms, ndx2, axis =0)
    
    distance = tf.subtract(pos2, pos1)
    distance = tf.square(distance)
    distance = tf.reduce_sum(distance, axis = 1)
    distance = tf.sqrt(distance)

    #to prevent nans
    distance = tf.clip_by_value(distance, 10E-4, 1000.0)
    
    r_6 = tf.pow(distance, 6)
    r_12 = tf.pow(r_6, 2)
    
    c6_term = tf.divide(c6, r_6)
    c12_term = tf.divide(c12, r_12)
    
    en = tf.subtract(c12_term, c6_term)

    en = tf.reduce_sum(en, axis = 0)
    return en


def tf_energy_lj_env(atoms_mol, atoms_env, atom_types):
    atoms_mol = tf.divide(atoms_mol, 10.0)
    atoms_env = tf.divide(atoms_env, 10.0)

    mask = tf.clip_by_value(atom_types, 0, 1)
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, 0) #(1,12)
    mask = tf.expand_dims(mask, 0) #(1,1,12)

    inv_mask = tf.subtract(mask, 1.0)
    inv_mask = tf.abs(inv_mask)
    #dummy_threshold = tf.multiply(inv_mask, 1e-10)
    
    atom_types = tf.subtract(atom_types, 1) #atom type indices were shifted by one to have dummy index at 0, so we have to subtract one here
    atom_types = tf.maximum(atom_types, 0) #push negative index back to 0 (they will be masked anyay later)
    atom_types = tf.expand_dims(atom_types, axis = 0)
    param_ndx = tf.range(0, n_atomtypes*n_atomtypes, n_atomtypes)
    param_ndx = tf.expand_dims(param_ndx, axis = 1)
    #make use of broadcasting
    param_ndx= tf.add(atom_types, param_ndx) #(4,12)
    
    param = tf.gather(params, param_ndx, axis = 0) #(4,12, 2)
    param = tf.expand_dims(param, 1) #(4,1,12,2)
   
    c6 = param[:,:,:,0] #(4,1,12)
    c12 = param[:,:,:,1]
    
    pos1 = tf.expand_dims(atoms_mol, axis = 0) #(1,12,3)
    pos1 = tf.expand_dims(pos1, axis = 0) #(1,1,12,3)
    pos2 = tf.expand_dims(atoms_env, axis = 2) #(4,10,1,3)
    
    distance = tf.subtract(pos1, pos2) #(4,10,12,3)
    distance = tf.square(distance)
    distance = tf.reduce_sum(distance, axis = 3) #(4,10,12)
    distance = tf.sqrt(distance)    

    #add 1 to all dummy distances to prevent zero division
    distance = tf.add(distance, inv_mask)

    #to prevent nans
    distance = tf.clip_by_value(distance, 10E-8, 1000.0)

    r6 = tf.pow(distance, 6)
    r12 = tf.pow(r6, 2)
    
         
    c6_term = tf.divide(c6, r6)
    c12_term = tf.divide(c12, r12)    
    
    en = tf.subtract(c12_term, c6_term)

    #multiply all potentials corresponding to dummie atoms with 0
    en = tf.multiply(en, mask)
    
    en = tf.reduce_sum(en, axis = [0,1,2])
    return en


"""
bond_constr = tf.constant(1000.0, tf.float32)

def tf_energy_bond_constr(atoms, indices):
    ndx1 = indices[1]
    ndx2 = indices[2]

    param_ndx = indices[0]
    param = tf.gather(params, param_ndx, axis = 0)
    
    a_0 = param[:,0]
    #f_c = bond_params[1] 

    pos1 = tf.gather(atoms, ndx1, axis =1)
    pos2 = tf.gather(atoms, ndx2, axis =1)
    
    dis = tf.subtract(pos1, pos2)
    dis = tf.square(dis)
    dis = tf.reduce_sum(dis, axis = 2)
    #dis = tf.sqrt(dis)
    #GROMACS USES NM WHILE THE DISTANCES HERE ARE GIVEN IN ANGSTROM
    dis = tf.divide(dis, 100.0)    

    dis_0 = tf.square(a_0)

    en = tf.subtract(dis, dis_0)
    en = tf.square(en)
    
    #dis2 = tf.cast(dis2, dtype = tf.float32)
    en = tf.multiply(en, bond_constr / 2.0)
    #en = tf.divide(en, 4.0)
    
    en = tf.reduce_sum(en, axis = 1)
    #e = tf.add(e, en)
    return en
"""

def tf_energy_angle_g96(atoms):
    ndx1 = angle_ndx[0]
    ndx2 = angle_ndx[1]
    ndx3 = angle_ndx[2]
    
    a_0 = angle_params[0]
    f_c = angle_params[1]

    
    pos1 = tf.gather(atoms, ndx1, axis =1)
    pos2 = tf.gather(atoms, ndx2, axis =1)
    pos3 = tf.gather(atoms, ndx3, axis =1)
    
    vec1 = tf.subtract(pos1, pos2)
    vec2 = tf.subtract(pos3, pos2)    

    norm1 = tf.square(vec1)
    norm1 = tf.reduce_sum(norm1, axis = 2)
    norm1 = tf.sqrt(norm1)    

    norm2 = tf.square(vec2)
    norm2 = tf.reduce_sum(norm2, axis = 2)
    norm2 = tf.sqrt(norm2)


    dot = tf.multiply(vec1, vec2)
    dot = tf.reduce_sum(dot, axis = 2)
    
    norm = tf.multiply(norm1, norm2)
    a = tf.divide(dot, norm)

    en = tf.subtract(a, a_0)
    en = tf.square(en)
    en = tf.multiply(en, f_c)
    en = tf.divide(en, 2.0)
    en = tf.reduce_sum(en, axis = 1)
    return en

def tf_energy_bonds_g96(atoms):
    ndx1 = bond_ndx[0]
    ndx2 = bond_ndx[1]
    
    a_0 = bond_params[0]
    f_c = bond_params[1] 

    pos1 = tf.gather(atoms, ndx1, axis =1)
    pos2 = tf.gather(atoms, ndx2, axis =1)
    
    dis = tf.subtract(pos1, pos2)
    dis = tf.square(dis)
    dis = tf.reduce_sum(dis, axis = 2)
    #dis = tf.sqrt(dis)
    #GROMACS USES NM WHILE THE DISTANCES HERE ARE GIVEN IN ANGSTROM
    dis = tf.divide(dis, 100.0)    

    dis_0 = tf.square(a_0)

    en = tf.subtract(dis, dis_0)
    en = tf.square(en)
    
    #dis2 = tf.cast(dis2, dtype = tf.float32)
    en = tf.multiply(en, f_c)
    en = tf.divide(en, 4.0)
    
    en = tf.reduce_sum(en, axis = 1)
    #e = tf.add(e, en)
    return en


def tf_energy_prop_dihedrals(atoms, indices):
    ndx1 = indices[1]
    ndx2 = indices[2]
    ndx3 = indices[3]
    ndx4 = indices[4]

    param_ndx = indices[0]
    param = tf.gather(params, param_ndx, axis = 0)
    
    a_0 = tf.multiply(param[:,0] , radiant_conversion)
    f_c = param[:,1]
    
    #a_0 = dihedrals_params[0]
    #f_c = dihedrals_params[1]
    #n = dihedrals_params[2]
    
    pos1 = tf.gather(atoms, ndx1, axis =0)
    pos2 = tf.gather(atoms, ndx2, axis =0)
    pos3 = tf.gather(atoms, ndx3, axis =0)
    pos4 = tf.gather(atoms, ndx4, axis =0)
    
    vec1 = tf.subtract(pos2, pos1)
    vec2 = tf.subtract(pos2, pos3)    
    vec3 = tf.subtract(pos4, pos3)    
    
    plane1 = tf.cross(vec1, vec2)
    plane2 = tf.cross(vec2, vec3)

    norm1 = tf.square(plane1)
    norm1 = tf.reduce_sum(norm1, axis = 1)
    norm1 = tf.sqrt(norm1)    

    norm2 = tf.square(plane2)
    norm2 = tf.reduce_sum(norm2, axis = 1)
    norm2 = tf.sqrt(norm2)


    dot = tf.multiply(plane1, plane2)
    dot = tf.reduce_sum(dot, axis = 1)
    
    norm = tf.multiply(norm1, norm2) + 1E-20
    a = tf.divide(dot, norm)
    
    a = tf.clip_by_value(a, -0.9999, 0.9999) #prevent nan because of rounding errors
    
    a = tf.acos(a)
    
    a = tf.multiply(a, 3.0)

    en = tf.subtract(a, a_0)
    en = tf.cos(en)
    en = tf.add(en, 1.0)
    en = tf.multiply(en, f_c)
    en = tf.reduce_sum(en, axis = 0)
    return en


def tf_energy_iprop_dihedrals(atoms, indices):
    ndx1 = indices[1]
    ndx2 = indices[2]
    ndx3 = indices[3]
    ndx4 = indices[4]

    param_ndx = indices[0]
    param = tf.gather(params, param_ndx, axis = 0)
    
    a_0 = tf.multiply(param[:,0] , radiant_conversion)
    f_c = param[:,1]
    
    #a_0 = dihedrals_params[0]
    #f_c = dihedrals_params[1]
    #n = dihedrals_params[2]
    
    pos1 = tf.gather(atoms, ndx1, axis =0)
    pos2 = tf.gather(atoms, ndx2, axis =0)
    pos3 = tf.gather(atoms, ndx3, axis =0)
    pos4 = tf.gather(atoms, ndx4, axis =0)
    
    vec1 = tf.subtract(pos2, pos1)
    vec2 = tf.subtract(pos2, pos3)    
    vec3 = tf.subtract(pos4, pos3)    
    
    plane1 = tf.cross(vec1, vec2)
    plane2 = tf.cross(vec2, vec3)

    norm1 = tf.square(plane1)
    norm1 = tf.reduce_sum(norm1, axis = 1)
    norm1 = tf.sqrt(norm1)    

    norm2 = tf.square(plane2)
    norm2 = tf.reduce_sum(norm2, axis = 1)
    norm2 = tf.sqrt(norm2)


    dot = tf.multiply(plane1, plane2)
    dot = tf.reduce_sum(dot, axis = 1)
    
    norm = tf.multiply(norm1, norm2) + 1E-20
    a = tf.divide(dot, norm)
    
    a = tf.clip_by_value(a, -0.9999, 0.9999) #prevent nan because of rounding errors
    
    a = tf.acos(a)
    
    #a = tf.multiply(a, 3.0)

    en = tf.subtract(a, a_0)
    #en = tf.cos(en)
    #en = tf.add(en, 1.0)
    en = tf.square(en)
    en = tf.multiply(en, f_c / 2.0)
    en = tf.reduce_sum(en, axis = 0)
    return en



def tf_energy_lj_14(atoms):
    ndx1 = pairs_ndx[0]
    ndx2 = pairs_ndx[1]

    
    c_6 = lj_14_params[0]
    c_12 = lj_14_params[1]

    
    pos1 = tf.divide(tf.gather(atoms, ndx1, axis =1), 10.0)
    pos2 = tf.divide(tf.gather(atoms, ndx2, axis =1), 10.0)
    
    distance = tf.subtract(pos2, pos1)
    distance = tf.square(distance)
    distance = tf.reduce_sum(distance, axis = 2)
    distance = tf.sqrt(distance)
    
    r_6 = tf.pow(distance, 6)
    r_12 = tf.pow(r_6, 2)
    
    c6_term = tf.divide(c_6, r_6)
    c12_term = tf.divide(c_12, r_12)
    
    en = tf.subtract(c12_term, c6_term)

    en = tf.reduce_sum(en, axis = 1)
    return en

def tf_energy_coulomb_14(atoms):
    ndx1 = pairs_ndx[0]
    ndx2 = pairs_ndx[1]

    #charge1 = charges[pairs[0]]
    #charge2 = charges[pairs[1]]

    charge1 = tf.gather(coulomb_14_params, ndx1, axis =0)
    charge2 = tf.gather(coulomb_14_params, ndx2, axis =0)

    pos1 = tf.divide(tf.gather(atoms, ndx1, axis =1), 10.0)
    pos2 = tf.divide(tf.gather(atoms, ndx2, axis =1), 10.0)
    
    distance = tf.subtract(pos2, pos1)
    distance = tf.square(distance)
    distance = tf.reduce_sum(distance, axis = 2)
    distance = tf.sqrt(distance)
       
    en = tf.multiply(charge1, charge2)
    en = tf.divide(en, distance)
    en = tf.multiply(en, 138.935458)

    en = tf.reduce_sum(en, axis = 1)
    return en


