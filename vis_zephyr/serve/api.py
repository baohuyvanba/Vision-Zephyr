# vis_zephyr/serve/api.py
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch

from vis_zephyr.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from vis_zephyr.conversation import templates
from vis_zephyr.model.builder import load_pretrained_model
from vis_zephyr.model.mm_utils import process_images, tokenizer_image_token

# Khởi tạo FastAPI app
app = FastAPI()

# --- 1 --- Load model, tokenizer, processor (chỉ chạy 1 lần khi server start)
print("[🚀] Loading model, tokenizer, processor...")

model_base = "HuggingFaceH4/zephyr-7b-beta"
model_path = "./checkpoints/vis-zephyr-7b-v1-pretrain"
model_name = "zephyr-7b-beta"  # Hoặc tự động lấy từ path

tokenizer, model, image_processor, context_length = load_pretrained_model(
    model_path = model_path,
    model_base = model_base,
    model_name = model_name,
    load_8bit  = False,
    load_4bit  = False,
    device_map = "auto",
    device     = "cuda"  # Hoặc "cpu" nếu test local
)

conv_template = templates["zephyr_v1"].copy()
roles = conv_template.roles

print("[✅] Model loaded and ready.")

# --- 2 --- Hàm decode base64 thành PIL Image
def decode_base64_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}")

# --- 3 --- API endpoint nhận POST
@app.post("/chat")
async def chat(request: Request):
    """
    Nhận JSON:
    {
      "image_base64": "<...>",
      "question": "..."
    }
    """
    try:
        data = await request.json()
        image_b64 = data.get("image_base64")
        question = data.get("question")

        if not image_b64 or not question:
            return JSONResponse({"error": "Missing image_base64 or question"}, status_code=400)

        # Decode ảnh
        pil_image = decode_base64_image(image_b64)

        # Preprocess ảnh
        image_tensor = process_images(
            images = pil_image,
            image_processor = image_processor,
            model_config = model.config
        )
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = [image_tensor.to(model.device, dtype=torch.float16)]

        # Tạo prompt
        user_input = DEFAULT_IMAGE_TOKEN + "\n" + question
        conversation = conv_template.copy()
        conversation.append_message(roles[0], user_input)
        conversation.append_message(roles[1], None)
        prompt = conversation.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        # Inference
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True
            )

        output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        return JSONResponse({"answer": output_text})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
