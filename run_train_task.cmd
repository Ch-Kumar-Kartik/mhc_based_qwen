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
set STEP2_OUT=output\qwen3_mhc_v2_parity_mixed_30k_1_latest

:model_found
echo [INFO] Step 1: HC-only training (3k steps) starting...
E:\mhc_based_qwen-2\qwvenv2\Scripts\python.exe -u train.py --tokenized-dir data/sangraha_packed/ --model output/qwen3_mhc_v2_converted_20260330_parity --base-model %BASE_MODEL% --output-dir %STEP1_OUT% --batch-size 1 --gradient-accumulation-steps 8 --total-steps 3000 --freeze-original-steps 3000 --learning-rate 1e-4 --warmup-steps 300 --save-steps 500 --eval-steps 500 --keep-last-checkpoints 3 --num-workers 0 --no-persistent-workers --prefetch-factor 1 --pin-memory --dtype auto --local-files-only --no-resume --log-interval 20 --train-only-hyper-connection-params 1>>E:\mhc_based_qwen-2\train.out_parity_hc_only_3k_1_latest.log 2>>E:\mhc_based_qwen-2\train.err_parity_hc_only_3k_1_latest.log
if errorlevel 1 (
	echo [ERROR] Step 1 HC-only training failed. Check train.err_parity_hc_only_3k_1_latest.log
	exit /b 1
)

if not exist "%STEP1_CKPT%\model.safetensors" (
	echo [ERROR] Step 1 checkpoint missing at %STEP1_CKPT%
	exit /b 2
)

echo [INFO] Step 2: Mixed LM+HC training (30k steps) starting from %STEP1_CKPT%...
E:\mhc_based_qwen-2\qwvenv2\Scripts\python.exe -u train.py --tokenized-dir data/sangraha_packed/ --model %STEP1_CKPT% --base-model %BASE_MODEL% --output-dir %STEP2_OUT% --batch-size 1 --gradient-accumulation-steps 8 --total-steps 30000 --freeze-original-steps 0 --learning-rate 1e-4 --warmup-steps 300 --save-steps 500 --eval-steps 500 --keep-last-checkpoints 3 --num-workers 0 --no-persistent-workers --prefetch-factor 1 --pin-memory --dtype auto --local-files-only --no-resume --log-interval 20 --use-mixed-lm-hc-loss --lm-loss-weight 0.75 --hc-loss-weight 0.25 1>>E:\mhc_based_qwen-2\train.out_parity_mixed_30k_1_latest.log 2>>E:\mhc_based_qwen-2\train.err_parity_mixed_30k_1_latest.log
