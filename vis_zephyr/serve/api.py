# vis_zephyr/serve/api.py
import base64
from io import BytesIO
from PIL import Image
from transformers import TextStreamer
import torch
from queue import Queue
import threading
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from vis_zephyr.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from vis_zephyr.conversation import templates, SeparatorStyle
from vis_zephyr.model.builder import load_pretrained_model
from vis_zephyr.model.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from vis_zephyr.model.multi_scale_process import process_any_resolution_image

# Khởi tạo FastAPI app
app = FastAPI()

# Bật CORS để cho phép frontend gọi API từ trình duyệt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Cho phép tất cả origin; thay bằng ["http://localhost:3000"] nếu muốn chỉ cho frontend local
    allow_credentials=True,
    allow_methods=["*"],            # Cho phép tất cả method: GET, POST, OPTIONS,...
    allow_headers=["*"],            # Cho phép tất cả header
)

# --- 1 --- Load model, tokenizer, processor (chỉ chạy 1 lần khi server start)
print("[🚀] Loading model, tokenizer, processor...")

model_base = "HuggingFaceH4/zephyr-7b-beta"
model_path = "./checkpoints/vis-zephyr-7b-v1-pretrain"
model_name = "zephyr-7b-beta"  # Hoặc tự động lấy từ path
sessions = {}

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

@app.post("/chat")
async def chat(request: Request):
    """
    Nhận JSON:
    {
      "session_id": "abc123",
      "image_base64": "<...>",  # Bắt buộc ở lần đầu
      "question": "..."
    }
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")
        question = data.get("question")
        image_b64 = data.get("image_base64")

        if not session_id or not question:
            return JSONResponse({"error": "Missing session_id or question"}, status_code=400)

        # Tạo mới conversation nếu chưa có
        if session_id not in sessions:
            conversation = templates["zephyr_v1"].copy()
            sessions[session_id] = {
                "conversation": conversation,
                "first_image_sent": False
            }
        else:
            conversation = sessions[session_id]["conversation"]

        # Lần đầu tiên: bắt buộc có ảnh
        if not sessions[session_id]["first_image_sent"]:
            if not image_b64:
                return JSONResponse({"error": "Missing image_base64 for first request"}, status_code=400)

            # Decode ảnh
            pil_image = decode_base64_image(image_b64)
            images_size = pil_image.size

            # Preprocess ảnh
            image_tensor = process_any_resolution_image(
                image=pil_image,
                processor=image_processor,
                grid_pinpoints=model.config.mm_grid_pinpoints,
            )
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = [image_tensor.to(model.device, dtype=torch.float16)]

            # Prepend image token
            if model.config.mm_use_im_start_end:
                user_input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
            else:
                user_input = DEFAULT_IMAGE_TOKEN + "\n" + question

            # Lưu lại cho session
            sessions[session_id]["first_image_sent"] = True
            sessions[session_id]["image_tensor"] = image_tensor
            sessions[session_id]["images_size"] = images_size

        else:
            # Những lần sau chỉ text
            user_input = question
            image_tensor = sessions[session_id]["image_tensor"]
            images_size = sessions[session_id]["images_size"]

        # Append user & assistant message
        conversation.append_message(conversation.roles[0], user_input)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        # Tokenize prompt
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        # Stopping criteria
        stop_str = conversation.separator_01 if conversation.separator_style == SeparatorStyle.ZEPHYR else conversation.separator_02
        stopping_criteria = KeywordsStoppingCriteria(
            keywords=[stop_str],
            tokenizer=tokenizer,
            input_ids=input_ids
        )

        async def generate_stream():
            q = Queue()

            class MyStreamer(TextStreamer):
                def on_finalized_text(self, text: str, stream_end: bool = False):
                    q.put(text)

            streamer = MyStreamer(
                tokenizer=tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            def _generate():
                with torch.inference_mode():
                    model.generate(
                        input_ids=input_ids,
                        images=image_tensor,
                        images_size=[images_size],
                        do_sample=True,
                        temperature=0.1,
                        max_new_tokens=256,
                        streamer=streamer,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria]
                    )
                q.put(None)  # Đánh dấu kết thúc

            # Chạy generate ở thread khác, để generate và yield chạy song song
            threading.Thread(target=_generate, daemon=True).start()

            while True:
                text = q.get()
                if text is None:
                    break
                yield text
        return StreamingResponse(generate_stream(), media_type="text/plain")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)