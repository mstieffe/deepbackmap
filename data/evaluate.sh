dir=./t313
bm=$dir/deepbackmap.xtc
aa=$dir/aa.xtc
em=$dir/baseline.xtc
mkdir ./evaluation
dir_data=./evaluation/t313
mkdir $dir_data
ndx_dir=./ndx_files

echo '0' > inp1
printf '0\n0' > inp2

#gmx rdf -f $bm -s ps_excl.tpr -o $save_dir/rdf_bm_all.xvg -n $ndx_dir/index_rdf_all.ndx -bin 0.01 -excl -xvg none  < inp2
#gmx rdf -f $aa -s ps_excl.tpr -o $save_dir/rdf_aa_all.xvg -n $ndx_dir/index_rdf_all.ndx -bin 0.01 -excl -xvg none  < inp2
#gmx rdf -f $em -s ps_excl.tpr -o $save_dir/rdf_em_all.xvg -n $ndx_dir/index_rdf_all.ndx -bin 0.01 -excl -xvg none  < inp2

#RDF
save_dir=$dir_data/rdf
mkdir $save_dir
gmx rdf -f $em -s ps_excl.tpr -o $save_dir/rdf_em.xvg -n $ndx_dir/index_rdf_carbon.ndx -bin 0.01 -excl -xvg none  < inp2
gmx rdf -f $bm -s ps_excl.tpr -o $save_dir/rdf_bm.xvg -n $ndx_dir/index_rdf_carbon.ndx -bin 0.01 -excl -xvg none  < inp2
gmx rdf -f $aa -s ps_excl.tpr -o $save_dir/rdf_aa.xvg -n $ndx_dir/index_rdf_carbon.ndx -bin 0.01 -excl -xvg none  < inp2

#Idih sc CCCC
save_dir=$dir_data/idih_sc_cccc
mkdir $save_dir
gmx angle -f $bm -n $ndx_dir/index_idih_sc_carbon.ndx -type improper -od $save_dir/idih_sc_bm.xvg -binwidth 2 -xvg none
gmx angle -f $aa -n $ndx_dir/index_idih_sc_carbon.ndx -type improper -od $save_dir/idih_sc_aa.xvg -binwidth 2 -xvg none
gmx angle -f $em -n $ndx_dir/index_idih_sc_carbon.ndx -type improper -od $save_dir/idih_sc_em.xvg -binwidth 2 -xvg none

#Pdih sc CCCC
save_dir=$dir_data/pdi_bb_cccc
mkdir $save_dir
gmx angle -f $bm -n $ndx_dir/index_pdih_bb_carbon.ndx -type dihedral -od $save_dir/pdih_bb_bm.xvg -binwidth 4 -xvg none
gmx angle -f $aa -n $ndx_dir/index_pdih_bb_carbon.ndx -type dihedral -od $save_dir/pdih_bb_aa.xvg -binwidth 4 -xvg none
gmx angle -f $em -n $ndx_dir/index_pdih_bb_carbon.ndx -type dihedral -od $save_dir/pdih_bb_em.xvg -binwidth 4 -xvg none

#Angle sc CCC
save_dir=$dir_data/angle_sc_ccc
mkdir $save_dir
gmx angle -f $bm -n $ndx_dir/index_angle_sc_carbon.ndx -od $save_dir/angle_sc_bm.xvg -xvg none
gmx angle -f $aa -n $ndx_dir/index_angle_sc_carbon.ndx -od $save_dir/angle_sc_aa.xvg -xvg none
gmx angle -f $em -n $ndx_dir/index_angle_sc_carbon.ndx -od $save_dir/angle_sc_em.xvg -xvg none

#Angle bb CCC
save_dir=$dir_data/angle_bb_ccc
mkdir $save_dir
gmx angle -f $bm -n $ndx_dir/index_angle_bb_carbon.ndx -od $save_dir/angle_bb_bm.xvg -binwidth 2 -xvg none
gmx angle -f $aa -n $ndx_dir/index_angle_bb_carbon.ndx -od $save_dir/angle_bb_aa.xvg -binwidth 2 -xvg none
gmx angle -f $em -n $ndx_dir/index_angle_bb_carbon.ndx -od $save_dir/angle_bb_em.xvg -binwidth 2 -xvg none

#Angle bb CCH
save_dir=$dir_data/angle_bb_cch
mkdir $save_dir
gmx angle -f $bm -n $ndx_dir/index_angle_bb_cch.ndx -od $save_dir/angle_bb_cch_bm.xvg -xvg none -binwidth 2
gmx angle -f $aa -n $ndx_dir/index_angle_bb_cch.ndx -od $save_dir/angle_bb_cch_aa.xvg -xvg none -binwidth 2
gmx angle -f $em -n $ndx_dir/index_angle_bb_cch.ndx -od $save_dir/angle_bb_cch_em.xvg -xvg none -binwidth 2

#Angle bb HCH
save_dir=$dir_data/angle_bb_hch
mkdir $save_dir
gmx angle -f $bm -n $ndx_dir/index_angle_bb_hch.ndx -od $save_dir/angle_bb_hch_bm.xvg -xvg none -binwidth 2
gmx angle -f $aa -n $ndx_dir/index_angle_bb_hch.ndx -od $save_dir/angle_bb_hch_aa.xvg -xvg none -binwidth 2
gmx angle -f $em -n $ndx_dir/index_angle_bb_hch.ndx -od $save_dir/angle_bb_hch_em.xvg -xvg none -binwidth 2

#Angle sc CCH
save_dir= $dir_data/angle_sc_cch
mkdir $save_dir
gmx angle -f $bm -n $ndx_dir/index_angle_sc_cch.ndx -od $save_dir/angle_sc_cch_bm.xvg -xvg none -binwidth 2
gmx angle -f $aa -n $ndx_dir/index_angle_sc_cch.ndx -od $save_dir/angle_sc_cch_aa.xvg -xvg none -binwidth 2
gmx angle -f $em -n $ndx_dir/index_angle_sc_cch.ndx -od $save_dir/angle_sc_cch_em.xvg -xvg none -binwidth 2

#Bonds bb CCC
save_dir=$dir_data/bond_bb_cc
mkdir $save_dir
gmx distance -f $bm -n $ndx_dir/index_dis_bb_carbon.ndx -oh $save_dir/dis_bb_bm.xvg -xvg none  < inp1
gmx distance -f $aa -n $ndx_dir/index_dis_bb_carbon.ndx -oh $save_dir/dis_bb_aa.xvg -xvg none  < inp1
gmx distance -f $em -n $ndx_dir/index_dis_bb_carbon.ndx -oh $save_dir/dis_bb_em.xvg -xvg none  < inp1

#Bonds bb CCC
save_dir=$dir_data/bond_sc_cc
mkdir $save_dir
gmx distance -f $bm -n $ndx_dir/index_dis_sc_carbon.ndx -oh $save_dir/dis_sc_bm.xvg -xvg none  < inp1
gmx distance -f $aa -n $ndx_dir/index_dis_sc_carbon.ndx -oh $save_dir/dis_sc_aa.xvg -xvg none  < inp1
gmx distance -f $em -n $ndx_dir/index_dis_sc_carbon.ndx -oh $save_dir/dis_sc_em.xvg -xvg none  < inp1


rm inp1
rm inp2
