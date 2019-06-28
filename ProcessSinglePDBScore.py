import math
import random
import glob
import os
import subprocess
import sys

pdb_id = sys.argv[1]
relaxed = eval(sys.argv[2])
out_dir = sys.argv[3]

if relaxed == True:
    out_dir += "%s/%s/%s/Relaxed"%(pdb_id[0], pdb_id[1], pdb_id)
    relaxed = "Relax"
else:
    out_dir += "%s/%s/%s/UnRelaxed"%(pdb_id[0], pdb_id[1], pdb_id)
    relaxed = ""
os.system("mkdir %s"%out_dir)
print(out_dir)

filename = "%s_%sOutput.txt"%(pdb_id, relaxed)
#expHP is the exposed hydrophobic surface area of FAMILYVW amino acids and therefore will not include others; printing to a separate file
expHP = subprocess.check_output(["sed", "-n", "/Begin report for expHP/,/End report for expHP/p;/End report for expHP/q", filename])
expHP = expHP.decode("utf-8").strip().split("\n")
expHP = [x.split("[0m")[-1] for x in expHP[2:-3]]
expHP = [x.split() for x in expHP]
if expHP[0][1] == "Resi":
    with open("%s/%s_%sExposedHydrophob.txt"%(out_dir, pdb_id, relaxed), "w+") as outData:
        for entry in expHP:
            #print(entry)
            outData.write(entry[2] + "\t" + entry[4][:-1] + "\n")
else:
    with open("%s/%s_%sExposedHydrophob.txt"%(out_dir, pdb_id, relaxed), "w+") as outData:
        for entry in expHP:
            #print(entry)
            outData.write(entry[1] + "\t" + entry[3][:-1] + "\n")

#dssp/bsa get appended to the byResScores    
dssp = subprocess.check_output(["grep", "protocols.DsspMover", filename])
dssp = dssp.decode("utf-8").strip().split("\n")[0]
dssp = list(dssp.split("\x1b")[-1].split("[0m")[-1])
if dssp[0] == "p":
    dssp = subprocess.check_output(["grep", "protocols.DsspMover", filename])
    dssp = dssp.decode("utf-8").strip().split("\n")[0]
    dssp = list(dssp.split(":")[-1].strip())

#buried surface area/dssp/rotamer probabilites and energy reductions are for each residue - including metals and can be added to the ByResScore file
bsa = subprocess.check_output(["sed", "-n", "/Begin report for BSA/,/TOTAL BURIED AREA/p;/TOTAL BURIED AREA/q", filename])
bsa = bsa.decode("utf-8").strip().split("\n")
bsa = [x.split("[0m")[-1].split("\t") for x in bsa[2:-1]]
if "REPORT" in bsa[0][0]:
    residues = [x[0].split()[-1] for x in bsa]
else:
    residues = [x[0] for x in bsa]
bsa = [x[1] for x in bsa]

if relaxed == "Relax":
    filename2 = "%s_ByResRelaxedScore.txt"%pdb_id
else:
    filename2 = "%s_ByResScore.txt"%pdb_id
by_res_terms = subprocess.check_output(["sed", "-n", "/^pose/,/END_POSE_ENERGIES/{/^pose/d;/END_POSE_ENERGIES/d;p;}", filename2])
by_res_terms = by_res_terms.decode("utf-8").strip().split("\n")
by_res_terms = ["\t".join(x.replace(":CtermProteinFull", "").replace(":NtermProteinFull", "").split()) for x in by_res_terms]
all_data = zip(by_res_terms, bsa, dssp)
all_data = ["\t".join(x) for x in all_data]
#for entry in all_data:
    #print(entry)

labels = subprocess.check_output(["grep", "label", filename2])
labels = labels.decode("utf-8").strip().split("\n")[0].split()
labels = "\t".join(labels) + "\tBSA\tDSSP\n"
with open("%s/%s_ByRes%sValues.txt"%(out_dir, pdb_id, relaxed), "w+") as outData:    
    outData.write(labels)
    outData.write("\n".join(all_data))
