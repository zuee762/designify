from flask import Flask, render_template, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import base64
import os
from io import BytesIO

app = Flask(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Running on: {device.upper()}")

pipe = StableDiffusionPipeline.from_pretrained(
    "dreamlike-art/dreamlike-diffusion-1.0",
    torch_dtype=dtype
)

pipe = pipe.to(device)

pipe.enable_attention_slicing()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/generate", methods=["POST"])
def generate():

    data = request.get_json()

    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({
            "error": "Prompt cannot be empty."
        }), 400

    try:

        with torch.no_grad():

            image = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]

        buffer = BytesIO()
        image.save(buffer, format="PNG")

        img_str = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            "image_url": f"data:image/png;base64,{img_str}"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )