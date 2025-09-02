import os
from pathlib import Path
os.environ['SPCONV_ALGO'] = 'native'

from io import BytesIO

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response
import uvicorn
import argparse
import torch

from omegaconf import OmegaConf
from loguru import logger
from src.model_utils import Model
from src.mvdream_utils import load_mvdream_model
import argparse
from PIL import Image
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
   
    return parser.parse_args()


args = get_args()
app = FastAPI()


model = load_mvdream_model(
    pretrained_model_name_or_path = args.model_name, 
    device = args.device
)
image_transform = None 





    

@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    #models: list = Depends(get_models),
) -> Response:
    text_input = str(prompt)
    num_of_frames = 4
    testing_views = [0, 6, 10, 26]

    images_np, image_views = model.inference_step(prompt=text_input, num_frames=num_of_frames, testing_views=testing_views)
    images = [Image.fromarray(image) for image in images_np]

    save_dir = Path(args.output_dir) / Path("mv_images")
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        output_path = os.path.join(save_dir, f"image_{i}.png")
        img.save(output_path, format = "PNG")

    return Response("", media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
    



