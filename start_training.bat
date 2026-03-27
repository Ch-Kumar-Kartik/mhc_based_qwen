@echo off
cd /d E:\mhc_based_qwen-2
set PYTHONUNBUFFERED=1
set TRANSFORMERS_VERBOSITY=debug
set TOKENIZERS_PARALLELISM=false
echo [%date% %time%] START>>launcher.log

E:\mhc_based_qwen-2\qwvenv2\Scripts\python.exe -u train.py ^
  --tokenized-dir data/sangraha_packed ^
  --base-model Qwen/Qwen3-0.6B ^
  --n-streams 4 --num-fracs 1 --sinkhorn-iters 20 ^
  --output-dir output/qwen3_mhc_v2 ^
  --batch-size 1 --gradient-accumulation-steps 8 ^
  --total-steps 50000 --learning-rate 1e-4 ^
  --save-steps 1000 --eval-steps 500 --keep-last-checkpoints 3 ^
  --num-workers 0 --no-persistent-workers --prefetch-factor 1 ^
  --pin-memory --dtype auto --local-files-only ^
  >> train.out.log 2>> train.err.log

echo [%date% %time%] EXIT_CODE=%ERRORLEVEL%>>launcher.log