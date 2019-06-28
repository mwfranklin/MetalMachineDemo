import os
import subprocess
import sys
import numpy as np
import scipy
import pandas as pd
import warnings
#personal libraries found in CustomModules
import PDBparser as pdbp
import PDBmanip as pdbm
import grid_tools
import pocket_lining
warnings.filterwarnings(action="ignore") #comment this out for making modifications

metal_size = {np.nan: 0,
         'MO':1, 'MOO':5, '4MO' : 1,'6MO' :1,'MOS': 4,
         'MG': 1,'3NI':1,'NI' : 1, 'ZN': 1,'MGF': 4,'MN3' : 1,'MN' : 1,'CO': 1,
         'OFE': 2, 'FE2': 1,'FEO': 3, 'FE' : 1,'FES': 4,
         'CU': 1, 'C2O' :3, 'CUA' : 2, 'CU1': 1, "3CO": 1,
         }

def collate_by_res_scores(pdb_id, protein, directory, relaxed):
    if relaxed == True:
        filename1 = "%s/%s_ByResRelaxValues.txt"%(directory, pdb_id)
        filename2 = "%s/%s_RelaxExposedHydrophob.txt"%(directory, pdb_id)
    else:
        filename1 = "%s/%s_ByResValues.txt"%(directory, pdb_id)
        filename2 = "%s/%s_ExposedHydrophob.txt"%(directory, pdb_id)
    by_res_scores = pd.read_table(filename1)
    res_ids = by_res_scores["label"].values
    res_ids = [x.split("_")[-1] for x in res_ids]
    by_res_scores["ResID"] = res_ids
    
    by_res_hydrophob = pd.read_table(filename2, header=None, names = ["ResID", "expHP"])
    res_ids = by_res_hydrophob["ResID"].values
    res_ids = [x[3:] for x in res_ids]
    by_res_hydrophob["ResID"] = res_ids
    by_res = pd.merge(by_res_scores, by_res_hydrophob, on="ResID", how = "outer")
    #print(by_res)
    #charmingly, rosetta's scores return with pdb-numbering not pose-numbering. 
    res_nums = []
    try:
        for x in by_res["ResID"].values:
            #print(x)
            res_nums.append(this_protein.res_nums[int(x)-1])
        by_res["ResID"] = res_nums #set ResID to be pose-numbering
        return(by_res)
    except IndexError:
        print("CHECK THIS SITE", pdb_id)
        return(by_res)    

def calculate_local_score(neighbor_list, score_df):
    near_res = score_df[score_df["ResID"].isin(neighbor_list)]
    #print(near_res)    
    local_score = near_res[beta_nov_terms].sum()
    local_score = local_score.add_suffix("_Sum")
    local_score2 = near_res[beta_nov_terms].mean()
    local_score2 = local_score2.add_suffix("_Mean")
    local_score = pd.concat([local_score, local_score2])
    #print(local_score)
    local_score["BSA"] = near_res["BSA"].sum()
    local_score["expHP"] = near_res["expHP"].sum()
    if len(near_res) != 0:
        local_score["LoopDSSP"] = len(near_res[near_res.DSSP == "L"])/len(near_res)
        local_score["HelixDSSP"] = len(near_res[near_res.DSSP == "H"])/len(near_res)
        local_score["SheetDSSP"] = len(near_res[near_res.DSSP == "E"])/len(near_res)
    else:
        local_score["LoopDSSP"] = 0
        local_score["HelixDSSP"] = 0
        local_score["SheetDSSP"] = 0
    return(local_score)

def calculate_local_score_by_shell(score_df, cutoff1, cutoff2):
    near_res = score_df[ (score_df["CBDist"] > cutoff1) & (score_df["CBDist"] <= cutoff2) ]
    #print(near_res)
    local_score = near_res[beta_nov_terms].sum()
    #print(type(local_score))
    local_score = local_score.add_suffix("_Sum")
    local_score2 = near_res[beta_nov_terms].mean()
    local_score2 = local_score2.add_suffix("_Mean")
    local_score = pd.concat([local_score, local_score2])

    local_score["BSA"] = near_res["BSA"].sum()
    local_score["expHP"] = near_res["expHP"].sum()
    if len(near_res) != 0:
        local_score["LoopDSSP"] = len(near_res[near_res.DSSP == "L"])/len(near_res)
        local_score["HelixDSSP"] = len(near_res[near_res.DSSP == "H"])/len(near_res)
        local_score["SheetDSSP"] = len(near_res[near_res.DSSP == "E"])/len(near_res)
    else:
        local_score["LoopDSSP"] = 0
        local_score["HelixDSSP"] = 0
        local_score["SheetDSSP"] = 0
    return(local_score)

def calculate_dist_to_atom_name(protein, atom_coords, atom_name):
    atom_type_coords = np.zeros((len(protein.residues), 3))
    atom_type_names = []
    for res in range(0, len(protein.residues)):
        this_res = protein.residues[res]
        #print(this_res.Atoms)
        atom_type_names.append(str(this_res.resnum) + this_res.chain)
        if this_res.type != "protein":
            atom_type_coords[res] = np.mean(this_res.Coords)
        elif this_res.name == "Gly": #Gly won't have CB
            try:
                atom_type_coords[res] = this_res.Coords[ this_res.Atoms.index(atom_name) ]
            except:
                atom_type_coords[res] = this_res.Coords[ this_res.Atoms.index("CA") ]
        else: #all other residues, including uncoded metals
            try:
                atom_type_coords[res] = this_res.Coords[ this_res.Atoms.index(atom_name) ]
            except:
                atom_type_coords[res] = np.mean(this_res.Coords)
    #print(atom_coords)
    #print(atom_type_names)
    all_dist = scipy.spatial.distance.cdist(atom_type_coords, [atom_coords]).flatten()
    d = {"ResID":np.asarray(atom_type_names), "%sDist"%atom_name:all_dist}
    distances = pd.DataFrame(data = d)
    #print(distances.head(10))
    return(distances)

def collate_findgeo(these_metals, pdb_id, directory):
    #print(these_metals)
    this_geom = np.zeros(49) #36 geom classifiers, irr/reg/distorted, 39 charge, 40-43, N/O/S/other ligands, 44 gRMSD, 45 max gRMSD dev, 46 valence, 47 nVESCUM, 48 overall RMSD
    #findgeo input is atomid, geom, irr/reg/distorted, RMSD, 4 N, O, S, other ligand, 8 charge, 9 gRMSD, 10 max dev w/in gRMSD, 11 valence, 12 nVESCUM
    with open("%s/%s.findgeo"%(directory, pdb_id), "r") as inData:
        for line in inData:
            line = line.strip().split("\t")
            #print(line)
            if len(line) == 13:
                ions = line[0].split("_")
                this_ion = ions[0] + "_" + ions[1] + ions[-1]
                line[0] = this_ion
                if this_ion in these_metals:
                    #print(line)
                    this_geom[ findgeo_geoms.index(line[1]) ] += 1#this_geom[0:36] for the 36 regular geometries + irregular
                    if line[2] == "regular": 
                        this_geom[37] += 1
                        this_geom[48] += float(line[3]) #overal RMSD
                        this_geom[44] += float(line[9]) #gRMSD
                        this_geom[45] += float(line[10]) #max gRMSD deviation
                    elif line[2] == "distorted": 
                        this_geom[38] += 1
                        this_geom[48] += float(line[3])
                        this_geom[44] += float(line[9])
                        this_geom[45] += float(line[10])
                    else:
                        #geom[36] counts irregular, so no need to calculate twice
                        this_geom[48] = np.nan
                        this_geom[44] = np.nan
                        this_geom[45] = np.nan
                    this_geom[40] += float(line[4]) #N
                    this_geom[41] += float(line[5])
                    this_geom[42] += float(line[6])
                    this_geom[43] += float(line[7])
                    this_geom[39] += float(line[8]) #charge
                    this_geom[46] += float(line[11]) #valence
                    this_geom[47] += float(line[12]) #nVESCUM
    these_labels = ["geom_" + x for x in findgeo_geoms]
    these_labels.extend(["geom_Reg", "geom_Distort", "geom_Charge", "geom_LigN", "geom_LigO", "geom_LigS", "geom_LigOther", "geom_gRMSD", "geom_MaxgRMSDDev", "geom_val", "geom_nVESCUM", "geom_AtomRMSD"])
    #print(this_geom, this_geom[36])
    this_geom = pd.DataFrame.from_records(this_geom.reshape(-1, len(this_geom)).T, index = these_labels)
    return(this_geom)

def extract_bluues_data(df, prefix):
    this_mean = df.mean(axis = 0, numeric_only = True)
    this_mean = this_mean[["mu2", "mu3", "mu4"]]
    this_mean = this_mean.add_prefix("Elec_%s_mean_"%prefix)
    this_max = df.max(axis = 0, numeric_only=True)
    this_max = this_max.add_prefix("Elec_%s_max_"%prefix)
    this_min = df.min(axis = 0, numeric_only = True)
    this_min = this_min[["pKa_shift", "dpKa_titr"]]
    this_min = this_min.add_prefix("Elec_%s_min_"%prefix)
    bluues_data = pd.concat([this_mean, this_min, this_max])
    #print(bluues_data)
    return(bluues_data)

def collate_bluues(pdb_id, directory, neighbor_list1, neighbor_list2, neighbors_9a):
    neighbors = [x[:-1] for x in neighbor_list1] #drop chain from numbers
    neighbors2 = set(neighbor_list2).difference(neighbor_list1)
    neighbors2 = [x[:-1] for x in neighbors2]
    #print(neighbors, neighbors2)
    titr_data = pd.read_csv("%s/%s_ElectFeatures.txt"%(directory, pdb_id), sep = "\t", header = 0)
    titr_data.rename(index = str, columns = {"Unnamed: 0":"ResID"}, inplace = True)
    titr_data.ResID = titr_data.ResID.str[4:]
    titr_data.drop(["dpKa_desolv", "dpKa_bg", "GBR6"], axis = 1, inplace=True)
    inside_neigh = titr_data[titr_data.ResID.isin(neighbors)]
    outside_neigh = titr_data[titr_data.ResID.isin(neighbors2)]
    inside = extract_bluues_data(inside_neigh, "ins")
    outside = extract_bluues_data(outside_neigh, "outs")
    neighbors_9a.ResID = neighbors_9a.ResID.str[:-1]
    #print(neighbors_9a)
    env_df = pd.merge(neighbors_9a, titr_data, on = "ResID", how = "inner").drop(["pKa_shift", "dpKa_titr"], axis = 1) #only keep titratable residues within 9A based on CA distance
    env_df["dist_weight"] = env_df.CADist.values
    env_df["dist_weight"] = 1/(env_df["dist_weight"]**2)
    env_df["mu2"] = env_df["mu2"]*env_df["dist_weight"]
    env_df["mu3"] = env_df["mu3"]*env_df["dist_weight"]
    env_df["mu4"] = env_df["mu4"]*env_df["dist_weight"]
    xenvr = env_df.sum(axis = 0, numeric_only = True)
    xenvr /= xenvr.dist_weight
    xenvr = xenvr[["mu2", "mu3", "mu4"]].add_prefix("Elec_xenv_")
    #print(xenvr)
    this_data = pd.concat([inside, outside, xenvr])
    #print(this_data)
    return(this_data)

#scoring terms for labeling    
whole_score_relax_terms = "total_score         BSA dslf_fa13     expHP    fa_atr fa_dun_dev fa_dun_rot fa_dun_semi     fa_elec fa_intra_atr_xover4 fa_intra_elec fa_intra_rep_xover4 fa_intra_sol_xover4              fa_rep              fa_sol hbond_bb_sc hbond_lr_bb    hbond_sc hbond_sr_bb    hxl_tors     lk_ball lk_ball_bridge lk_ball_bridge_uncpl lk_ball_iso       omega     p_aa_pp pro_close rama_prepro         ref        rmsd"
whole_score_relax_terms = whole_score_relax_terms.split()
#beta_nov2016 scoring terms and weights
beta_nov_terms = "fa_atr fa_rep fa_sol fa_intra_atr_xover4 fa_intra_rep_xover4 fa_intra_sol_xover4 lk_ball lk_ball_iso lk_ball_bridge lk_ball_bridge_uncpl fa_elec fa_intra_elec pro_close hbond_sr_bb hbond_lr_bb hbond_bb_sc hbond_sc dslf_fa13 omega fa_dun_dev fa_dun_rot fa_dun_semi p_aa_pp hxl_tors ref rama_prepro"
beta_nov_terms = beta_nov_terms.split()
beta_nov_weights = "1 0.55 1 1 0.55 1 0.92 -0.38 -0.33 -0.33 1 1 1.25 1 1 1 1 1.25 0.48 0.69 0.76 0.78 0.61 1 1 0.5"
beta_nov_weights = beta_nov_weights.split()
beta_nov_weights = [float(x) for x in beta_nov_weights]

findgeo_geoms = ["lin", "trv", "tri", "tev", "spv", 
    "tet", "spl", "bva", "bvp", "pyv", 
    "spy", "tbp", "tpv", 
    "oct", "tpr", "pva", "pvp", "cof", "con", "ctf", "ctn",
    "pbp", "coc", "ctp", "hva", "hvp", "cuv", "sav",
    "hbp", "cub", "sqa", "boc", "bts", "btt", 
    "ttp", "csa", "irr"]

relax = eval(sys.argv[1])
#print(relax, eval(relax))
if relax == True:
    relax_state = "Relaxed"
else:
    relax_state = "UnRelaxed"

#print(relax, relax_state)
sites = pd.read_csv("SITECenters_%s.txt"%relax_state, sep = '\t')
bad_pdb_codes = [1000, 500, 999]

#iterate through each site and extract info
cutoff1 = 3.5
cutoff2 = 5
cutoff3 = 7.5
cutoff4 = 10
whole_protein_data = pd.DataFrame()
for entry in sites.itertuples():
    #print(entry)
    pdb_id = entry.pdb_name
    site_id = entry.SITE_ID
    #root_dir = "/panfs/pfs.local/work/slusky/MSEAL/data/PDB_chains/%s/%s/%s/%s"%(pdb_id[0], pdb_id[1], pdb_id, relax_state)
    root_dir = "SampleData/%s/%s"%(pdb_id, relax_state)
    if ((os.path.isfile("%s/%s_Relaxed.pdb"%(root_dir, pdb_id)) == False) & (relax == True)): #did Rosetta finish?
        print("No pdb file", pdb_id, site_id)
        continue
    elif entry.X in bad_pdb_codes:
        print("Bad SITE code", pdb_id, site_id)
        continue
    elif entry.NearNeigh == "NotBound":
        print("Not bound", pdb_id, site_id)
        continue
    elif ( (os.path.isfile("%s/%s_ByResRelaxValues.txt"%(root_dir, pdb_id)) == False) & (relax == True) ):
        print("Not scored in Rosetta", pdb_id, site_id)
        continue
    elif ((os.path.isfile("%s/%s_ByResValues.txt"%(root_dir, pdb_id)) == False) & (relax == False) ):
        print("Not scored in Rosetta", pdb_id, site_id)
        continue        
    elif os.path.isfile("%s/%s_Grid.pdb"%(root_dir, site_id)) == False: #did the pocket grid get created?
        print("No grid file", pdb_id, site_id)
        continue
    elif os.path.isfile("%s/%s_ElectFeatures.txt"%(root_dir, pdb_id)) == False: #is there bluues electrostatics data?
        print("No electro file", pdb_id, site_id)
        continue 
    elif os.path.isfile("%s/%s.findgeo"%(root_dir, pdb_id)) == False: #is there findgeo/CMM data?
        print("No geom file", pdb_id, site_id)
        continue
    else:
        print(pdb_id, site_id)
        chain = entry.chainID
        if relax == True:
            pdb_file = "%s_Relaxed.pdb"%pdb_id
        else:
            pdb_file = "%s.pdb"%pdb_id
        residues, res_nums, header = pdbp.create_res("%s/%s"%(root_dir, pdb_file))
        #print(res_nums)
        this_protein = pdbp.Protein(residues, res_nums, header)
        center_of_mass = np.mean(this_protein.Coords, axis = 0)
        #print(center_of_mass)
        metals = set([str(entry.seqNum1), str(entry.seqNum2), str(entry.seqNum3), str(entry.seqNum4)])
        metal_res = []
        for x in metals:
            if x != "0": metal_res.append(str(round(float(x))) + chain )
        metals = [entry.resName1, entry.resName2, entry.resName3, entry.resName4]
        
        site_atom_count = 0
        for x in metals:
            if x != "0": site_atom_count += metal_size[x]
        
        #general SITE info
        SITE_info = pd.DataFrame([entry.SITE_ID, entry.catalytic, entry.ValidSet, entry.ECOD_arch, len(metal_res), site_atom_count], index=["SITE_ID", "Catalytic", "ValidSet", "ECOD_arch", "MetalCodes", "MetalAtoms"])
        #print(SITE_info)

        #coordination geometry
        findgeo_metals = []
        for x in zip([entry.resName1, entry.resName2, entry.resName3, entry.resName4], [str(entry.seqNum1), str(entry.seqNum2), str(entry.seqNum3), str(entry.seqNum4)]):
            if x[0] is not np.nan:
                findgeo_metals.append(x[0] + "_" + x[1] + chain)
        geom_feat = collate_findgeo(findgeo_metals, pdb_id, root_dir)
        #print(geom_feat)
        
        #set up for distance-dependent terms
        SITE_center_pt = np.asarray([entry.X, entry.Y, entry.Z])
                
        neighbors1 = sorted(set(this_protein.get_neighbor_res(SITE_center_pt, cutoff1)).difference(metal_res))
        neighbors2 = sorted(set(this_protein.get_neighbor_res(SITE_center_pt, cutoff2)).difference(metal_res))
        neighbors3 = sorted(set(this_protein.get_neighbor_res(SITE_center_pt, cutoff3)).difference(metal_res))
        neighbors4 = sorted(set(this_protein.get_neighbor_res(SITE_center_pt, cutoff4)).difference(metal_res))
        #print(neighbors1, neighbors2, neighbors3)
        #test_neigh = [x[:-1] for x in neighbors3]
        #print(":".join(test_neigh))
        
        #electrostatics calculations
        distances_9a = calculate_dist_to_atom_name(this_protein, SITE_center_pt, "CA")
        distances_9a = distances_9a[ (distances_9a.CADist <= 9.0) ]
        #print(distances_9a)
        bluues_electro = collate_bluues(pdb_id, root_dir, neighbors1, sorted(set(this_protein.get_neighbor_res(SITE_center_pt, 9)).difference(metal_res)), distances_9a)
        
        #calculate rosetta spheres
        by_res_relax_scores = collate_by_res_scores(pdb_id, this_protein, root_dir, relax)
        #print(by_res_relax_scores.tail(10))
        local_relax_score1 = calculate_local_score(neighbors1, by_res_relax_scores)
        local_relax_score1 = local_relax_score1.add_suffix("_"+str(cutoff1))
        local_relax_score2 = calculate_local_score(neighbors2, by_res_relax_scores)
        local_relax_score2 = local_relax_score2.add_suffix("_"+str(cutoff2))
        local_relax_score3 = calculate_local_score(neighbors3, by_res_relax_scores)
        local_relax_score3 = local_relax_score3.add_suffix("_"+str(cutoff3))
        local_relax_score4 = calculate_local_score(neighbors4, by_res_relax_scores)
        local_relax_score4 = local_relax_score4.add_suffix("_"+str(cutoff4))

        #calculate rosetta spheres
        cb_dist = calculate_dist_to_atom_name(this_protein, SITE_center_pt, "CB")
        by_res_shell_scores = pd.merge(by_res_relax_scores, cb_dist, on = "ResID")
        shell1 = calculate_local_score_by_shell(by_res_shell_scores, cutoff1, cutoff2)
        shell1 = shell1.add_suffix("_S"+str(cutoff2)) 
        shell2 = calculate_local_score_by_shell(by_res_shell_scores, cutoff2, cutoff3)
        shell2 = shell2.add_suffix("_S"+str(cutoff3))
        shell3 = calculate_local_score_by_shell(by_res_shell_scores, cutoff3, cutoff4)
        shell3 = shell3.add_suffix("_S"+str(cutoff4))

        #differeniate backbone/sidechain neighbors
        nearest_neigh_coords = grid_tools.closest_residue(SITE_center_pt, this_protein.residues[ res_nums.index(int(entry.NearNeigh[:-1])) ].Coords)
        pocket_grid = grid_tools.process_pocket("%s/%s_Grid.pdb"%(root_dir, site_id))
        whole_pocket = np.asarray(pocket_grid[-1])
        #print(whole_pocket)
        if len(whole_pocket) == 0:
            print("No pocket", pdb_id, site_id)
        if len(whole_pocket) > 0:
            print("Full pocket", pdb_id, site_id)
            clusters = pocket_lining.cluster_grid_points(whole_pocket)
            #print(clusters)
            whole_pocket = pocket_lining.get_nearby_grid( whole_pocket, clusters, SITE_center_pt, 5)
            #print(whole_pocket)
        
            pocket_lining_res = pocket_lining.id_pocket_lining(whole_pocket, this_protein, cutoff=3)
            pocket_lining_res = set(pocket_lining_res).difference(metal_res)
            #print(pocket_lining_res, res_nums)
            pocket_lining_res = [x for x in pocket_lining_res if this_protein.residues[ this_protein.res_nums.index(x) ].type == "protein"]
            pymol_pocket = [x[:-1] for x in pocket_lining_res]
            #print("+".join(pymol_pocket))
            #pocket_lining.graph_grid(np.asarray(pocket_grid[-1]), clusters, SITE_center_pt)
            bb, bb_names, sc, sc_names = pocket_lining.split_pocket_lining(pocket_lining_res, whole_pocket, this_protein, cutoff = 2.2)
            #print(bb, bb_names, sc, sc_names)
        else:
            bb = []
            bb_names = []
            sc = []
            sc_names = []
        #calculate lining and pocket features
        #print(bb_names, sc_names)
        labels, pocket_lining_list = pocket_lining.calc_lining_features(bb_names, sc_names, this_protein)
        #print(labels)
        pocket_lining_list = pd.DataFrame(pocket_lining_list, index=labels) 
        
        dist_to_center = np.linalg.norm(SITE_center_pt - center_of_mass)
        max_dist_to_center = np.sqrt(np.max( np.sum( (this_protein.Coords - dist_to_center)**2, axis = 1) ))
        labels, pocket_feature_list = grid_tools.calc_pocket_features(pdb_id, entry.SITE_ID, pocket_grid, SITE_center_pt, nearest_neigh_coords, center_of_mass, directory = root_dir)
        labels.extend(["SITEDistCenter", "SITEDistNormCenter"])
        pocket_feature_list.extend([dist_to_center, dist_to_center/max_dist_to_center ])
        pocket_feature_list = pd.DataFrame(pocket_feature_list, index=labels)
        #print(pocket_feature_list)
        
        #dump the whole thing into a single dataframe and append to growing list
        if len(whole_pocket > 0):
            whole_protein = pd.concat([SITE_info, local_relax_score1, local_relax_score2, local_relax_score3, local_relax_score4, shell1, shell2, shell3, pocket_feature_list, pocket_lining_list, bluues_electro, geom_feat]).T
            whole_protein_data = whole_protein_data.append(whole_protein, ignore_index=True)

whole_protein_data["NoSC_vol"] = whole_protein_data["Vol"] + whole_protein_data["occ_vol"]
whole_protein_data["SC_vol_perc"] = whole_protein_data["occ_vol"] / whole_protein_data["NoSC_vol"]
bad_terms = ("hbond_lr_", 'dslf_fa13', 'pro_close', 'total')
whole_protein_data.drop(columns = [term for term in whole_protein_data if term.startswith(bad_terms)], inplace = True)

score_weights = dict(zip(beta_nov_terms, beta_nov_weights))
if relax == True:
    #deal with weighted scores compared to the unrelaxed scores
    #Not sure this is important because we scale all the weights eventually, but it does make generating graphs of raw data later much easier
    all_feature_names = list(whole_protein_data)
    for entry in score_weights.keys():
        this_weight = score_weights[entry]
        these_pos = [i for i, s in enumerate(all_feature_names) if entry in s]
        #print(entry, these_pos)
        for idx in these_pos:
            whole_protein_data.iloc[:,idx] /= this_weight

#print(list(whole_protein_data))
#print(len(list(whole_protein_data)))
#print(whole_protein_data["NoSC_vol"], whole_protein_data["Vol"], whole_protein_data["occ_vol"])
#print(whole_protein_data)
whole_protein_data.to_csv("WholeScores_%s.txt"%relax_state, sep="\t", index=False)