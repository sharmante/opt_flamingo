import torch
import time
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from PIL import Image
import requests

# ✅ 장치 설정 (CUDA 사용 가능 여부 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로드 (FP32 상태)
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
    )

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
model.to(device) # FP32 상태로 GPU에 로드

# ✅ 샘플 이미지 & 텍스트 준비
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# 이미지 전처리 (FP32)
vision_x = image_processor(image).unsqueeze(0).unsqueeze(1).unsqueeze(0).to(device)


# 텍스트 전처리 (input_ids는 Long 유지)
tokenizer.padding_side = "left"
lang_x = tokenizer(["<image>An image of a cat.<|endofchunk|><image>An image of"], return_tensors="pt").to(device)
lang_x["input_ids"] = lang_x["input_ids"].long() # ✅ input_ids는 반드시 LongTensor 유지
lang_x["attention_mask"] = lang_x["attention_mask"].to(torch.bfloat16) # ✅ attention_mask는 BF16 변환 가능

# ✅ 추론 및 성능 측정 함수
def benchmark_model(model, vision_x, lang_x):
    torch.cuda.empty_cache() # 캐시 정리
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated() # 메모리 측정 시작
    start_time = time.time() # 시간 측정 시작

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=20,
        num_beams=3,
        )

    torch.cuda.synchronize()
    end_time = time.time() # 시간 측정 종료
    end_mem = torch.cuda.memory_allocated() # 메모리 측정 종료

    output_text = tokenizer.decode(generated_text[0]) # 생성된 텍스트
    inference_time = end_time - start_time # 추론 시간 계산
    memory_usage = (end_mem - start_mem) / (1024**2) # MB 단위로 변환

    return output_text, inference_time, memory_usage

# ✅ FP32 모델 성능 측정
output_fp32, time_fp32, mem_fp32 = benchmark_model(model, vision_x, lang_x)

# ✅ BF16 변환 후 재처리
vision_x = image_processor(image).unsqueeze(0).unsqueeze(1).unsqueeze(0).to(device, dtype=torch.bfloat16)
lang_x["attention_mask"] = lang_x["attention_mask"].to(torch.bfloat16) # ✅ attention_mask만 BF16 변환

# ✅ 모델을 BF16으로 변환
model.to(torch.bfloat16)

# ✅ BF16 모델 성능 측정
output_bf16, time_bf16, mem_bf16 = benchmark_model(model, vision_x, lang_x)

# ✅ 결과 출력
print("\n=== FP32 vs BF16 성능 비교 ===")
print(f"FP32 추론 속도: {time_fp32:.4f} 초, 메모리 사용량: {mem_fp32:.2f} MB")
print(f"BF16 추론 속도: {time_bf16:.4f} 초, 메모리 사용량: {mem_bf16:.2f} MB")
print("\n=== 생성된 텍스트 비교 ===")
print(f"FP32 Output: {output_fp32}")
print(f"BF16 Output: {output_bf16}")

# ✅ 성능 개선률 계산
speedup = (time_fp32 - time_bf16) / time_fp32 * 100
mem_reduction = (mem_fp32 - mem_bf16) / mem_fp32 * 100

print("\n=== 성능 개선률 ===")
print(f"추론 속도 향상: {speedup:.2f}%")
print(f"메모리 절약: {mem_reduction:.2f}%")