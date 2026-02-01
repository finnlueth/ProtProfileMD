"""
This script generates a Foldseek database from an amino-acid FASTA file and a corresponding 3Di FASTA file.
It reads both files, checks for matching sequence IDs, and writes temporary TSV files for amino-acid sequences,
3Di strings, and headers. It then calls Foldseek to create the database files and cleans up the temporary files.
Usage: python generate_foldseek_db.py <in_fasta> <in_3di> <db_name> <tmp_dir>
"""

import sys
import os
from Bio import SeqIO

in_fasta = sys.argv[1]
in_3di = sys.argv[2]
db_name = sys.argv[3]
tmp_dir = sys.argv[4]

# read in amino-acid sequences
sequences_aa = {}
for record in SeqIO.parse(in_fasta, "fasta"):
    sequences_aa[record.id] = str(record.seq)

# read in 3Di strings
sequences_3di = {}
for record in SeqIO.parse(in_3di, "fasta"):
    if not record.id in sequences_aa.keys():
        print("Warning: ignoring 3Di entry {}, since it is not in the amino-acid FASTA file".format(record.id))
    else:
        sequences_3di[record.id] = str(record.seq).upper()

# assert that we parsed 3Di strings for all sequences in the amino-acid FASTA file
for id in sequences_aa.keys():
    if not id in sequences_3di.keys():
        print("Error: entry {} in amino-acid FASTA file has no corresponding 3Di string".format(id))
        quit()

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(db_name)), exist_ok=True)

# generate TSV file contents
with open(f"{tmp_dir}/aa.tsv", "w") as f_aa, \
    open(f"{tmp_dir}/3di.tsv", "w") as f_3di, \
    open(f"{tmp_dir}/header.tsv", "w") as f_header, \
    open(f"{db_name}.lookup", "w") as f_lookup:
    for i, id in enumerate(sequences_aa.keys(), start=1):
        f_aa.write(f"{i}\t{sequences_aa[id]}\n")
        f_3di.write(f"{i}\t{sequences_3di[id]}\n")
        f_header.write(f"{i}\t{id}\n")
        f_lookup.write(f"{i}\t{id}\t0\n")


# create Foldseek database
os.system("foldseek tsv2db {}/aa.tsv {} --output-dbtype 0".format(tmp_dir, db_name))
os.system("foldseek tsv2db {}/3di.tsv {}_ss --output-dbtype 0".format(tmp_dir, db_name))
os.system("foldseek tsv2db {}/header.tsv {}_h --output-dbtype 12".format(tmp_dir, db_name))

# clean up
os.remove("{}/aa.tsv".format(tmp_dir))
os.remove("{}/3di.tsv".format(tmp_dir))
os.remove("{}/header.tsv".format(tmp_dir))
