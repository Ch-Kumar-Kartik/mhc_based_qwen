from src.conversionV2 import convert_qwen3_to_mhc_v2

# Load original model and convert
mhc_model = convert_qwen3_to_mhc_v2(
    model_name_or_path="Qwen/Qwen3-0.6B",
    output_path="./qwen3-0.6b-mhc"
)