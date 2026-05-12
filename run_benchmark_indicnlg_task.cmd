@echo off
cd /d E:\mhc_based_qwen-2
set PYTHONUNBUFFERED=1
set TRANSFORMERS_VERBOSITY=debug
set TOKENIZERS_PARALLELISM=false
if not defined HF_TOKEN set HF_TOKEN=
if not defined HUGGING_FACE_HUB_TOKEN set HUGGING_FACE_HUB_TOKEN=%HF_TOKEN%
set HF_HOME=E:\mhc_based_qwen-2\hf_home
set HF_HUB_CACHE=E:\mhc_based_qwen-2\hf_cache
set TRANSFORMERS_CACHE=E:\mhc_based_qwen-2\hf_cache
set HUGGINGFACE_HUB_CACHE=E:\mhc_based_qwen-2\hf_cache
set HF_HUB_OFFLINE=0
set TRANSFORMERS_OFFLINE=0
set HF_DATASETS_OFFLINE=0

echo [INFO] Running IndicNLG benchmark for Qwen3 mHC V2 checkpoint...
E:\mhc_based_qwen-2\qwvenv2\Scripts\python.exe -u -m scripts.benchmark_indicnlg --model output\qwen3_mhc_v2_parity_mixed_30k_1_latest\checkpoint-30000 --model-type mhc-v2 --base-model Qwen/Qwen3-0.6B --split test --apply-chat-template --max-samples-per-language 0 --enable-tf32 --enable-cudnn-benchmark 1>E:\mhc_based_qwen-2\benchmark_indicnlg.out.log 2>E:\mhc_based_qwen-2\benchmark_indicnlg.err.log