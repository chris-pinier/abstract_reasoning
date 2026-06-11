#!/bin/bash

#SBATCH --job-name=ar_exp1
#SBATCH --partition=genoa
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --output=slurm_output/SLURM-%x-%A_%a.out
#SBATCH --error=slurm_output/SLURM-%x-%A_%a.err
#
# For job-array use, uncomment/edit this line. Example: subjects 1-40, max 8 concurrent jobs.
##SBATCH --array=1-40%8

set -euo pipefail

echo "=== SLURM ALLOCATION ==="
scontrol show job "$SLURM_JOB_ID" \
    | grep -E "TRES|MinMemory|QOS|Command|NumCPUs|NumNodes|Partition|JobName|Array" \
    | tr ' ' '\n' \
    | grep -v "^$" \
    || true
echo "========================"

# Avoid CPU oversubscription when running multiple subject jobs in parallel.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# -----------------------------
# User-configurable paths
# -----------------------------
# Override these at submission time if needed:
#   sbatch --export=ALL,PROJECT_DIR=/path/to/repo,DATA_ROOT=/path/to/data,OUTPUT_BASE_DIR=/path/to/scratch ...
PROJECT_DIR="${PROJECT_DIR:-$HOME/abstract_reasoning}"
DATA_ROOT="${DATA_ROOT:-/path/to/experiment1/data}"
HUMAN_DATA_DIR="${HUMAN_DATA_DIR:-$DATA_ROOT/Lab/raw-BIDS3}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$DATA_ROOT/Lab/preprocessed}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/scratch-shared/$USER/abstract_reasoning/experiments}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/analysis/.venv}"

# SUBJECTS_CSV can be:
#   - "all" to let Python discover all subjects
#   - "1,2,3" to run multiple subjects in one job
#   - with a SLURM array: "1,2,3,4,5" and each task picks one subject
SUBJECTS_CSV="${SUBJECTS_CSV:-all}"

RNG_SEED="${RNG_SEED:-0}"
N_WORKERS="${N_WORKERS:-1}"
CHANNEL_GROUP="${CHANNEL_GROUP:-frontal}"
FRP_BASELINE="${FRP_BASELINE:-none}"
RESPONSE_BASELINE="${RESPONSE_BASELINE:-none}"
REST_BASELINE="${REST_BASELINE:-none}"
FRP_CONTROL_METHOD="${FRP_CONTROL_METHOD:-circular_shift}"
RANDOM_CONTROL_START_EVENT="${RANDOM_CONTROL_START_EVENT:-stim-all_stim}"
RANDOM_CONTROL_END_EVENT="${RANDOM_CONTROL_END_EVENT:-trial_end}"
RANDOM_CONTROL_MIN_ONSET_DIFF_S="${RANDOM_CONTROL_MIN_ONSET_DIFF_S:-0.300}"
SEQUENCE_HEATMAP_METRIC="${SEQUENCE_HEATMAP_METRIC:-count}"

mkdir -p slurm_output
mkdir -p "$OUTPUT_BASE_DIR"

echo "Job started at: $(date)"
echo "PROJECT_DIR:      $PROJECT_DIR"
echo "DATA_ROOT:        $DATA_ROOT"
echo "HUMAN_DATA_DIR:   $HUMAN_DATA_DIR"
echo "PREPROCESSED_DIR: $PREPROCESSED_DIR"
echo "OUTPUT_BASE_DIR:  $OUTPUT_BASE_DIR"
echo "SUBJECTS_CSV:     $SUBJECTS_CSV"
echo "N_WORKERS:        $N_WORKERS"
echo "CHANNEL_GROUP:    $CHANNEL_GROUP"
echo "FRP_BASELINE:     $FRP_BASELINE"
echo "RESPONSE_BASELINE:$RESPONSE_BASELINE"
echo "REST_BASELINE:    $REST_BASELINE"
echo "FRP_CONTROL:      $FRP_CONTROL_METHOD"
echo "HEATMAP_METRIC:   $SEQUENCE_HEATMAP_METRIC"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "--------------------------------------"

cd "$PROJECT_DIR/analysis"

if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
fi

# If this is an array job and SUBJECTS_CSV is a list, select one subject per task.
# If SUBJECTS_CSV=all and this is an array job, the task id itself is used as the subject number.
SUBJ_NS="$SUBJECTS_CSV"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ "$SUBJECTS_CSV" == "all" ]]; then
        SUBJ_NS="$SLURM_ARRAY_TASK_ID"
    else
        IFS=',' read -r -a SUBJECT_ARRAY <<< "$SUBJECTS_CSV"
        ARRAY_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
        if [[ "$ARRAY_INDEX" -lt 0 || "$ARRAY_INDEX" -ge "${#SUBJECT_ARRAY[@]}" ]]; then
            echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID is outside SUBJECTS_CSV length=${#SUBJECT_ARRAY[@]}"
            exit 1
        fi
        SUBJ_NS="${SUBJECT_ARRAY[$ARRAY_INDEX]}"
    fi
fi

echo "Running subject selection: $SUBJ_NS"

start_time=$(date +%s)

srun python scripts/experiment1_pipeline.py \
    --data-root "$DATA_ROOT" \
    --human-data-dir "$HUMAN_DATA_DIR" \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --output-base-dir "$OUTPUT_BASE_DIR" \
    --subj-ns "$SUBJ_NS" \
    --rng-seed "$RNG_SEED" \
    --n-workers "$N_WORKERS" \
    --channel-group "$CHANNEL_GROUP" \
    --frp-baseline "$FRP_BASELINE" \
    --response-baseline "$RESPONSE_BASELINE" \
    --rest-baseline "$REST_BASELINE" \
    --frp-control-method "$FRP_CONTROL_METHOD" \
    --random-control-start-event "$RANDOM_CONTROL_START_EVENT" \
    --random-control-end-event "$RANDOM_CONTROL_END_EVENT" \
    --random-control-min-onset-diff-s "$RANDOM_CONTROL_MIN_ONSET_DIFF_S" \
    --sequence-heatmap-metric "$SEQUENCE_HEATMAP_METRIC"

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "Job finished at: $(date)"
echo "Total runtime: $((runtime / 60)) minutes and $((runtime % 60)) seconds."
