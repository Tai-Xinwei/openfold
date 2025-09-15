#!/usr/bin/env bash
set -euo pipefail


FASTA_DIR="/project-abd/xinwei/msaprocess/casp15_fasta"
MMCIF_DIR="/project-abd/xinwei/msaprocess/mmcif"
CASP_LMDB="/project-abd/xinwei/msaprocess/proteintest-casp-14and15-targets.20250814_3ea28a39.lmdb"
REORGANIZE_SCRIPT="/project-abd/xinwei/msaprocess/reorgan_openfold_output.py"    # 确保该脚本在当前目录或写绝对路径
EVALUATE_SCRIPT="/project-abd/xinwei/msaprocess/evaluation_jianwei/EvaluateCASP4Openfold.py"           # 确保该脚本在当前目录或写绝对路径
CONFIG_PRESET="model_3"
##need to change
OUTPUT_DIR="/project-abd/xinwei/msaprocess/predict/casp15_nmask_1_64_randomdenoise_p50_k10"
PRECOMP_MSA="/project-abd/xinwei/msaprocess/MSA_generated/casp15.nmask_1_64_randomdenoise_p50_k10"
MODEL_DEVICE="cuda:0"


LOG_DIR="${OUTPUT_DIR%/}/logs"
mkdir -p "${LOG_DIR}"

# ==== 跑 OpenFold 推理 ====
echo "[$(date '+%F %T')] Start OpenFold inference..."
python3 run_pretrained_openfold.py \
  "${FASTA_DIR}" \
  "${MMCIF_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_precomputed_alignments "${PRECOMP_MSA}" \
  --config_preset "${CONFIG_PRESET}" \
  --model_device "${MODEL_DEVICE}" \
  --skip_relaxation \
  | tee "${LOG_DIR}/openfold_infer_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] OpenFold finished."

# ==== 调用整理脚本，把 OUTPUT_DIR 传进去 ====
if [[ ! -f "${REORGANIZE_SCRIPT}" ]]; then
  echo "[ERROR] ${REORGANIZE_SCRIPT} 不存在，请检查路径。" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Reorganizing predictions under: ${OUTPUT_DIR}"
python3 "${REORGANIZE_SCRIPT}" "${OUTPUT_DIR}"

echo "[$(date '+%F %T')] Reorganization finished."

# ==== 调用 evaluation.py 做评测 ====
if [[ ! -f "${EVALUATE_SCRIPT}" ]]; then
  echo "[ERROR] ${EVALUATE_SCRIPT} 不存在，请检查路径。" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Start evaluation..."
python3 "${EVALUATE_SCRIPT}" "${CASP_LMDB}" "${OUTPUT_DIR}" \
  | tee "${LOG_DIR}/evaluation_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date '+%F %T')] All done."
