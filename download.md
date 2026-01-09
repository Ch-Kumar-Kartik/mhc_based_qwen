# Download and tokenize 10,000 samples
python download.py --num_samples 10000 --output_dir ./data
# Download with custom settings
python download.py --num_samples 100000 --max_length 4096 --output_dir ./data
# Use with training
python -m scripts.train \
  --model ./qwen3-0.6b-mhc \
  --data ./data/fineweb_edu_qwen3 \
  --config configs/qwen3_0.6b_mhc.yaml