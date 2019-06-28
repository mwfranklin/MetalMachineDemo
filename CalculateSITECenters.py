import os
import PDBparser as pdbp
import PDBmanip as pdbm
import numpy as np
import pandas as pd
import sys

def SITE_center(metal_resIDs, myProtein):
    #metal_resIDs should already be a in the format of a string of ResChain ie 395A
    center_coords = []
    #print(myProtein.res_nums)
    #print(metal_resIDs)
    metal_res = [i for i, e in enumerate(myProtein.res_nums) if e in metal_resIDs]
    #print(metal_res)
    if len(metal_res) == 0:
        print("BAD PDB")
        test_coords = [999,999,999]
    elif len(metal_res) != len(metal_resIDs):
        print("MISSING ATOMS")
        test_coords = [500,500,500]
    else:
        these_metal_res = []
        for x in metal_res:
            center_coords.extend(myProtein.residues[x].Coords)
        center_coords = np.row_stack(center_coords)
        test_coords = np.mean(center_coords, axis = 0)
    return(test_coords)

def nearest_neigh(site_center, metal_resIDs, myProtein):
    distances, near_res = myProtein.KDTree.query(site_center, k = 40, distance_upper_bound = 10)
    distances = [x for x in distances if np.isinf(x) == False]
    if len(distances) == 0:
        return("NA")
        
    near_res = near_res[0:len(distances)]
    new_res_ids = [this_protein.Coords_index[x] for x in near_res]
    new_res_ids = sorted(set(new_res_ids), key = new_res_ids.index)
    new_res_ids = [x for x in new_res_ids if x not in metal_resIDs]
    #print(new_res_ids)
    if len(new_res_ids) == 0:
        return("NotBound")
    else:
        for x in range(0, len(new_res_ids)):
            this_res = myProtein.residues[ myProtein.res_nums.index(new_res_ids[x]) ]
            #print(new_res_ids[x], this_res.resnum, this_res.type)
            if this_res.type == "protein":
                return(new_res_ids[x])
        return("NotBound")

relaxed = sys.argv[1]
site_list = pd.read_csv("SITECenters_%s.txt"%relaxed, sep = "\t", header = 0)#change to reading in from SITECenters later
site_list["X"].fillna(1000, inplace = True)
site_list["Y"].fillna(1000, inplace = True)
site_list["Z"].fillna(1000, inplace = True)
site_list["NearNeigh"].fillna("na", inplace = True)
site_list.fillna(0, inplace = True)

#iterate through each site and extract info
for entry in site_list.itertuples():
    pdb_id = entry.pdb_name
    #print(entry)
    if entry.Y in [1000,999,500]:
        #print("Uncalculated PDB", entry)
            
        metals = [entry.seqNum1, entry.seqNum2, entry.seqNum3, entry.seqNum4]
        metals = set([ int(x) for x in list([entry.seqNum1, entry.seqNum2, entry.seqNum3, entry.seqNum4]) if x != 0 ])
        #print(pdb_id, metals)
        #site_list.loc[entry.Index, "X"] = 19
    
        if relaxed == "Relaxed":
            pdb_id = pdb_id + "_Relaxed"
        if os.path.isfile("/panfs/pfs.local/work/slusky/MSEAL/data/PDB_chains/%s/%s/%s/%s/%s.pdb"%(pdb_id[0], pdb_id[1], pdb_id[0:6], relaxed, pdb_id) ) == False: #pdb_id[0], pdb_id[1], 
            #if os.path.isfile("SampleData/%s/UnRelaxed/%s.pdb"%(pdb_id, pdb_id) ) == False: #pdb_id[0], pdb_id[1], 
            continue
        else:
            print(pdb_id, metals)
            chain = pdb_id[5]
            metal_res = [str(x) + chain for x in metals ]
            #residues, res_nums, header = pdbp.create_res("SampleData/%s/UnRelaxed/%s.pdb"%(pdb_id, pdb_id)) #test on local
            residues, res_nums, header = pdbp.create_res("/panfs/pfs.local/work/slusky/MSEAL/data/PDB_chains/%s/%s/%s/%s/%s.pdb"%(pdb_id[0], pdb_id[1],pdb_id[0:6], relaxed, pdb_id)) #for production run on CRC
            #print(res_nums)
            this_protein = pdbp.Protein(residues, res_nums, header)        
            SITE_center_pt = SITE_center(metal_res, this_protein)
            index = entry.Index
            site_list.loc[index,"X"] = SITE_center_pt[0]
            site_list.loc[index,"Y"] = SITE_center_pt[1]
            site_list.loc[index,"Z"] = SITE_center_pt[2]
            site_list.loc[index,"NearNeigh"] = nearest_neigh(SITE_center_pt, metal_res, this_protein)
            #print(site_list.loc[index,:])

site_list.to_csv("SITECenters_%s.txt"%relaxed, sep="\t", index=False)




