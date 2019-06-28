#! /bin/bash
#SBATCH --job-name=Pocket
#SBATCH --mail-user=mwfranklin@ku.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=6:00:00
#SBATCH --output=/panfs/pfs.local/work/slusky/MSEAL/logs/Pocket_%A_%a.log
#SBATCH --chdir=/panfs/pfs.local/work/slusky/MSEAL/data/
#SBATCH --partition=sixhour

line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" GridList.txt)
IFS=$'\t' read -r -a arr -d '' < <(printf '%s' "$line")
pdb_id=${arr[0]}
relaxed=${arr[1]}
site_id=${arr[2]}
rosetta_res=${arr[3]} #res:chain format for rosetta

cd PDB_chains/${pdb_id:0:1}/${pdb_id:1:1}/${pdb_id:0:6}/${relaxed} #check this filepath structure
$ROSETTA_DEV/pocket_measure.linuxgccrelease -s ${pdb_id}.pdb -central_relax_pdb_num ${rosetta_res} -pocket_num_angles 100 -pocket_dump_pdbs -pocket_filter_by_exemplar 1 -pocket_grid_size 15 -ignore_unrecognized_res > StdOutputPocket.txt

mv pocket0.pdb ${site_id}_Grid.pdb

sed -ne '/Pocket score/,$p' StdOutputPocket.txt > ${site_id}_Vol.txt

rm StdOutputPocket.txt
