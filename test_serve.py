import os
import open3d as o3d
from pathlib import Path
os.environ['SPCONV_ALGO'] = 'native'

from io import BytesIO

import torch
from src.dataset_utils import (
    get_singleview_data,
    get_multiview_data,
    get_voxel_data_json,
    get_image_transform_latent_model,
    get_pointcloud_data,
    get_mv_dm_data,
    get_sv_dm_data,
    get_sketch_data
)
from pytorch_lightning import seed_everything
from src.model_utils import Model
from src.mvdream_utils import load_mvdream_model
from PIL import Image
import time




model = load_mvdream_model(
    pretrained_model_name_or_path = "ADSKAILab/WaLa-MVDream-RGB4", 
    device = "cuda"
)
image_transform_ = None 
model_3d = Model.from_pretrained("ADSKAILab/WaLa-RGB4-1B")
image_transform_3d = get_image_transform_latent_model()



def simplify_mesh(obj_path, target_num_faces=1000):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    simplified_mesh = mesh.simplify_quadric_decimation(target_num_faces)
    o3d.io.write_triangle_mesh(obj_path, simplified_mesh)

def generate_3d_object(
    model,
    data,
    data_idx,
    scale,
    diffusion_rescale_timestep,
    save_dir="./",
    output_format="obj",
    target_num_faces=None,
    seed=42,
):
    # Set seed
    seed_everything(seed, workers=True)

    # save_dir.mkdir(parents=True, exist_ok=True)
    # image_name = save_dir.stem
    image_name = "test"
    model.set_inference_fusion_params(scale, diffusion_rescale_timestep)
    output_path = model.test_inference(
        data, data_idx,image_name, save_dir=save_dir, output_format=output_format
    )

    if output_format == "obj" and target_num_faces:
        simplify_mesh(output_path, target_num_faces=target_num_faces)




prompts_file = open("/workspace/vol_sub17/prompts.txt", "r")
cnt = 0
while cnt < 2 :
    torch.cuda.empty_cache()
    prompt = prompts_file.readline()

    text_input = str(prompt)
    num_of_frames = 4
    testing_views = [0, 6, 10, 26]

    images_np, image_views = model.inference_step(prompt=text_input, num_frames=num_of_frames, testing_views=testing_views)
    images = [Image.fromarray(image) for image in images_np]

    save_dir = '/workspace/vol_sub17/test-wala'
    # save_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        output_path = os.path.join(save_dir, f"image_{i}.png")
        img.save(output_path, format = "PNG")
    multiview_images = [
        f'{save_dir}/image_0.png',
        f'{save_dir}/image_1.png',
        f'{save_dir}/image_2.png',
        f'{save_dir}/image_3.png',
    ]
    image_views = ['image_0.png', 'image_1.png', 'image_2.png', 'image_3.png']
    data = get_multiview_data(
        image_files=multiview_images,
        views=image_views,
        image_transform=image_transform_3d,
        device=model_3d.device,
    )
    data_idx = 0
    save_dir = f'{save_dir}/result/'
    generate_3d_object(
        model_3d,
        data,
        data_idx,
        1.3,
        10,
        save_dir,
        "obj",
        None,
        42,
    )