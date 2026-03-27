@echo off
cd /d E:\mhc_based_qwen-2
set PYTHONUNBUFFERED=1
set TRANSFORMERS_VERBOSITY=debug
set TOKENIZERS_PARALLELISM=false
HF_HOME=E:\mhc_based_qwen-2\hf_home
set HF_HUB_CACHE=E:\mhc_based_qwen-2\hf_cache
set TRANSFORMERS_CACHE=E:\mhc_based_qwen-2\hf_cache
set HUGGINGFACE_HUB_CACHE=E:\mhc_based_qwen-2\hf_cache
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

set BASE_MODEL=
for /d %%D in ("E:\mhc_based_qwen-2\hf_cache\models--Qwen--Qwen3-0.6B\snapshots\*") do (
  set BASE_MODEL=%%~fD
  goto :model_found
)

echo [ERROR] Offline base model snapshot not found at E:\mhc_based_qwen-2\hf_cache\models--Qwen--Qwen3-0.6B\snapshots 1>>E:\mhc_based_qwen-2\train.err.log
exit /b 2

:model_found
E:\mhc_based_qwen-2\qwvenv2\Scripts\python.exe -u train.py --tokenized-dir data/sangraha_packed --base-model "%BASE_MODEL%" --n-streams 4 --num-fracs 1 --sinkhorn-iters 20 --output-dir output/qwen3_mhc_v2 --batch-size 1 --gradient-accumulation-steps 8 --total-steps 50000 --learning-rate 1e-4 --save-steps 1000 --eval-steps 500 --keep-last-checkpoints 3 --num-workers 0 --no-persistent-workers --prefetch-factor 1 --pin-memory --dtype auto --local-files-only 1>>E:\mhc_based_qwen-2\train.out.log 2>>E:\mhc_based_qwen-2\train.err.log
