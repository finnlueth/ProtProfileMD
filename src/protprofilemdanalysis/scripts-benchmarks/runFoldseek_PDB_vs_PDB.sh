#!/bin/bash

FOLDSEEK="foldseek"
BENCHMARK="bench.noselfhit"

FOLDSEEK_ANALYSIS_SCOP_DIR="./src/protprofilemdanalysis/foldseek-analysis/scopbenchmark"
PDB_DIR=./tmp/data/benchmarks/scop40pdb
OUT_DIR=./tmp/testing/pdb_vs_pdb

LOOKUP_FILE=${FOLDSEEK_ANALYSIS_SCOP_DIR}/data/scop_lookup.fix.tsv
TEMP_DIR=${OUT_DIR}/temp
ALIGN_DIR_RAW=${TEMP_DIR}/alignResults/rawoutput
ALIGN_DIR_TMP=${TEMP_DIR}/alignResults/tmp
ALIGN_DIR_ROCX=${TEMP_DIR}/


mkdir -p ${ALIGN_DIR_RAW}
mkdir -p ${ALIGN_DIR_TMP}
mkdir -p ${ALIGN_DIR_ROCX}


${FOLDSEEK} easy-search  ${PDB_DIR}/ ${PDB_DIR}/ ${ALIGN_DIR_RAW}/foldseekaln ${ALIGN_DIR_TMP}/ --threads 8 -s 9.5 --max-seqs 2000 -e 10 --sort-by-structure-bits 0
${FOLDSEEK_ANALYSIS_SCOP_DIR}/scripts/${BENCHMARK}.awk ${LOOKUP_FILE} <(cat ${ALIGN_DIR_RAW}/foldseekaln) > ${OUT_DIR}/foldseek.rocx

awk '{ famsum+=$3; supfamsum+=$4; foldsum+=$5}END{print famsum/NR,supfamsum/NR,foldsum/NR}' ${OUT_DIR}/foldseek.rocx | tee ${OUT_DIR}/foldseek.txt
