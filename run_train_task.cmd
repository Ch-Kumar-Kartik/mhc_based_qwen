@echo off
cd /d E:\mhc_based_qwen-2
set PYTHONUNBUFFERED=1
set TRANSFORMERS_VERBOSITY=debug
set TOKENIZERS_PARALLELISM=false
set HF_HOME=E:\mhc_based_qwen-2\hf_home
set HF_HUB_CACHE=E:\mhc_based_qwen-2\hf_cache
set TRANSFORMERS_CACHE=E:\mhc_based_qwen-2\hf_cache
set HUGGINGFACE_HUB_CACHE=E:\mhc_based_qwen-2\hf_cache
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1

set BASE_MODEL=output\qwen3_mhc_v2_converted_20260330_parity
set STEP1_OUT=output\qwen3_mhc_v2_parity_hc_only_3k_1_latest
set STEP1_CKPT=%STEP1_OUT%\checkpoint-3000
set STEP2_OUT=output\qwen3_mhc_v2_parity_mixed_100k_1_latest
set STEP2_RESUME=E:\mhc_based_qwen-2\output\qwen3_mhc_v2_parity_mixed_30k_1_latest\checkpoint-30000


:model_found
echo [INFO] Step 1: skipped.
if not exist "%STEP2_RESUME%\model.safetensors" (
	echo [ERROR] Resume checkpoint missing at %STEP2_RESUME%
	exit /b 2
)

echo [INFO] Step 2: Mixed LM+HC training (continuing from 30k checkpoint to 100k total steps) resuming from %STEP2_RESUME%...
uv run --python E:\mhc_based_qwen-2\mhc-qwen\Scripts\python.exe train.py --tokenized-dir data/sangraha_packed/ --model %STEP2_RESUME% --base-model %BASE_MODEL% --output-dir %STEP2_OUT% --batch-size 1 --gradient-accumulation-steps 8 --total-steps 100000 --freeze-original-steps 0 --learning-rate 1e-4 --warmup-steps 300 --save-steps 500 --eval-steps 500 --keep-last-checkpoints 3 --num-workers 0 --no-persistent-workers --prefetch-factor 1 --pin-memory --dtype auto --local-files-only --resume-from-checkpoint %STEP2_RESUME% --log-interval 20 --use-mixed-lm-hc-loss --lm-loss-weight 0.75 --hc-loss-weight 0.25 1>>E:\mhc_based_qwen-2\train.out_parity_mixed_100k_1_latest.log 2>>E:\mhc_based_qwen-2\train.err_parity_mixed_100k_1_latest.log