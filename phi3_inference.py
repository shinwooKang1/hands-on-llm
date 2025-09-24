import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Check available devices
print("=== M4 Chip Optimization ===")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# For M4, we can use both CPU and GPU strategically
if torch.backends.mps.is_available():
    device = "mps"  # Use GPU for model inference
    cpu_device = "cpu"  # Use CPU for preprocessing
    print("✅ Using M4 unified memory architecture")
    print("✅ GPU (MPS) for model inference")
    print("✅ CPU for data preprocessing")
else:
    device = "cpu"
    cpu_device = "cpu"
    print("⚠️  Using CPU only")

print(f"Main device: {device}")
print(f"CPU device: {cpu_device}")

# Load model and tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map=device,
    dtype=torch.float16,  # Use float16 for better performance on MPS
    trust_remote_code=False,
)

# Create a pipeline (without device argument for accelerate-loaded models)
print("Creating pipeline...")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,
    do_sample=False,
)

# Example of using both CPU and GPU strategically
prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

print(f"\nGenerating text for prompt: {prompt}")

# CPU에서 토큰화 (빠른 전처리)
print("🔄 Tokenizing on CPU...")
inputs = tokenizer(prompt, return_tensors="pt")

# GPU로 이동 (통합 메모리로 빠른 전송)
print("🚀 Moving to GPU...")
if device == "mps":
    inputs = {k: v.to(device) for k, v in inputs.items()}

# GPU에서 추론
print("⚡ Running inference on GPU...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# CPU로 다시 이동하여 디코딩
print("🔄 Decoding on CPU...")
if device == "mps":
    outputs = outputs.to(cpu_device)

generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("\n=== Generated Text ===")
print(generated_text)

print("\n=== Performance Summary ===")
print("✅ Utilized M4 unified memory architecture")
print("✅ CPU-GPU collaboration for optimal performance")
