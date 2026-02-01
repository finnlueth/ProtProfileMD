#!/bin/bash

"""
bash ./src/protprofilemdanalysis/scripts-benchmarks/runFoldseek_Profile_vs_3Di.sh \
--scope-bench-dir ./src/protprofilemdanalysis/foldseek-analysis/scopbenchmark \
--aa-fasta ./tmp/data/benchmarks/scope40_sequences_AA.fasta \
--3di-fasta ./tmp/runs/ProtProfileMD_20251113_072928_large_batch_size_again/benchmarks/scope_40_benchmark/scope40_3Di_argmax_without_background.fasta \
--profiles ./tmp/runs/ProtProfileMD_20251113_072928_large_batch_size_again/benchmarks/scope_40_benchmark/scope40_profiles_without_background_ratio.tsv \
--data-script-dir ./src/protprofilemdanalysis/scripts-data \
--out-dir ./tmp/testing/profile_vs_3di
"""

FOLDSEEK_ANALYSIS_SCOP_DIR=""
SEQUENCE_FASTA_AA=""
SEQUENCE_FASTA_3Di=""
PROFILE_CSV=""
DATA_SCRIPT_DIR=""
OUT_DIR=""

usage() {
    cat << EOF
Usage: $0 -d SCOP_DIR -a AA_FASTA -t 3DI_FASTA -p PROFILE -b RUN_NAME

Runs FoldSeek profile vs 3Di benchmark.

REQUIRED OPTIONS:
    -d, --scope-bench-dir DIR   Path to SCOP benchmark directory
    -a, --aa-fasta FILE         Path to amino acid FASTA file
    -t, --3di-fasta FILE        Path to 3Di FASTA file
    -p, --profiles FILE         Path to profile CSV file
    -s, --data-script-dir DIR   Path to data script directory
    -o, --out-dir DIR           Output directory

OTHER OPTIONS:
    -h, --help                  Show this help message

BACKGROUND EXECUTION:
    nohup $0 [OPTIONS] > ./results/my_benchmark/log.out &
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--scope-bench-dir)
            FOLDSEEK_ANALYSIS_SCOP_DIR="$2"
            shift 2
            ;;
        -a|--aa-fasta)
            SEQUENCE_FASTA_AA="$2"
            shift 2
            ;;
        -t|--3di-fasta)
            SEQUENCE_FASTA_3Di="$2"
            shift 2
            ;;
        -p|--profiles)
            PROFILE_CSV="$2"
            shift 2
            ;;
        -s|--data-script-dir)
            DATA_SCRIPT_DIR="$2"
            shift 2
            ;;
        -o|--out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

FOLDSEEK="foldseek"
BENCHMARK="bench.noselfhit"

LOOKUP_FILE=${FOLDSEEK_ANALYSIS_SCOP_DIR}/data/scop_lookup.fix.tsv

# mapped to /tmp. change if needed.
# TEMP_DIR=/tmp/${OUT_DIR}/temp
TEMP_DIR=${OUT_DIR}/temp
RESULT_DIR=${OUT_DIR}/
ALIGN_DIR_ROCX=${RESULT_DIR}
ALIGN_DIR_RAW=${TEMP_DIR}/alignResults/rawoutput
ALIGN_DIR_TMP=${TEMP_DIR}/alignResults/tmp
ALIGN_DIR_ALN=${TEMP_DIR}/alignResults/tmp/aln
DB_SEQUENCE=${TEMP_DIR}/alignResults/db_sequence
DB_PROFILE=${TEMP_DIR}/alignResults/db_profile
DB_NAME_FOLDSEEK=foldseekDB

mkdir -p ${ALIGN_DIR_ROCX}
mkdir -p ${ALIGN_DIR_RAW}
mkdir -p ${ALIGN_DIR_TMP} 
mkdir -p ${DB_SEQUENCE}
mkdir -p ${DB_PROFILE}

date

echo "Running SCOPe40 benchmark"

echo "Generating FoldSeek DB"
python ${DATA_SCRIPT_DIR}/generate_foldseek_db.py ${SEQUENCE_FASTA_AA} ${SEQUENCE_FASTA_3Di} ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${TEMP_DIR}

echo "Building Profile DB"
python ${DATA_SCRIPT_DIR}/build_profiledb.py ${PROFILE_CSV} ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${DB_PROFILE}

# foldseek makepaddedseqdb ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK}_padded

echo "Running FoldSeek search"
${FOLDSEEK} search ${DB_PROFILE}/${DB_NAME_FOLDSEEK}_profile ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_TMP} -s 9.5 --max-seqs 2000 -e 10
# ${FOLDSEEK} search ${DB_PROFILE}/${DB_NAME_FOLDSEEK}_profile ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK}_padded ${ALIGN_DIR_ALN} ${ALIGN_DIR_TMP} --gpu 1 -s 9.5 --max-seqs 2000 -e 10

echo "Converting FoldSeek alignments"
${FOLDSEEK} convertalis ${DB_PROFILE}/${DB_NAME_FOLDSEEK}_profile ${DB_SEQUENCE}/${DB_NAME_FOLDSEEK} ${ALIGN_DIR_ALN} ${ALIGN_DIR_RAW}/foldseekaln

date

${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/foldseekaln) > ${ALIGN_DIR_ROCX}/foldseek.rocx

awk '{famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ${ALIGN_DIR_ROCX}/foldseek.rocx | tee ${ALIGN_DIR_ROCX}/foldseek.txt
