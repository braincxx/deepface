
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
#MULTI (work)
from PIL import Image

import numpy as np
import mediapipe as mp
from controlnet_aux import OpenposeDetector, CannyDetector

from gfpgan import GFPGANer

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, DDIMScheduler, ControlNetModel, AutoencoderKL, StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy
from PIL import Image

from diffusers.pipelines.controlnet import MultiControlNetModel
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

from basicsr.utils import imwrite
import os




def create_face_mask(seg_mask, image_size):
    """
    Creates a binary mask from face landmarks for inpainting.
    
    Args:
        seg_mask (dict): Dictionary containing face detection results.
        image_size (tuple): Tuple (height, width) of the image.
        
    Returns:
        numpy.ndarray: Binary mask where the face region is white.
    """

    landmarks = seg_mask['landmark_2d_106']
    # Convert landmarks to integer coordinates for OpenCV
    landmarks_int = landmarks.astype(np.int32)
    
    # Create a blank mask of the same size as the image
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw the filled polygon of the face
  #  cv2.fillPoly(mask, [landmarks_int], 255)

    hull = cv2.convexHull(np.array(landmarks_int))
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    
    return Image.fromarray(mask)


def _prepare_control_images(tgt) -> list:
    """Подготовка контрольных изображений"""
    pose_processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    canny_processor = CannyDetector()
    return [
        canny_processor(tgt, threshold=(30, 100)),
        pose_processor(tgt, include_face=True)
    ]

def _align_faces(src_img: np.ndarray, tgt_img: np.ndarray, src_o, tgt_o) -> np.ndarray:
    # Extract landmarks
    src_landmarks = src_o['landmark_2d_106']
    tgt_landmarks = tgt_o['landmark_2d_106']
    
    if src_landmarks is None or tgt_landmarks is None:
        return src_img
    
    # Convert landmarks to integer coordinates
    src_landmarks_int = src_landmarks.astype(np.int32)
    tgt_landmarks_int = tgt_landmarks.astype(np.int32)
    
    # Calculate transformation matrix
    transformation_matrix, _ = cv2.estimateAffinePartial2D(
        src_landmarks_int, tgt_landmarks_int, method=cv2.LMEDS
    )
    
    if transformation_matrix is None:
        return src_img
    
    # Apply affine transformation
    transformed_image = cv2.warpAffine(
        src_img,
        transformation_matrix,
        (src_img.shape[1], src_img.shape[0]),  # Ensure correct dimensions
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC
    )
    
    # If the result is empty, create a copy of the source image
    if transformed_image is None or transformed_image.size == 0:
        transformed_image = src_img.copy()
    
    return transformed_image

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

path_source = "./source_image/1.jpg"
path_target= "./target_image/3.jpg"
path_save = f"./result_image/{os.path.basename(path_source)}_{os.path.basename(path_target)}.jpg"

image = cv2.imread(path_source)


faces = app.get(image)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face



target = cv2.imread(path_target)
target_face = app.get(target)
target_face_image = face_align.norm_crop(target, landmark=target_face[0].kps, image_size=224) 



# control_images = _prepare_control_images(target) + [Image.fromarray(_align_faces(face_image, target_face_image, faces[0], target_face[0]))]
# aligned_image = _align_faces(face_image, target_face_image, faces[0], target_face[0])
# target_face_image = aligned_image


mask = create_face_mask(target_face[0], (target.shape[0], target.shape[1]))
target = Image.open(path_target)


controlnets = []
controlnet_types = ["canny", "openpose", "depth", "face_landmarks"]
for cn_type in controlnet_types:
    controlnet = ControlNetModel.from_pretrained(
        f"lllyasviel/sd-controlnet-{cn_type}" if cn_type != "face_landmarks" 
        else "CrucibleAI/ControlNetMediaPipeFace",
        torch_dtype=torch.float16
    )
    controlnets.append(controlnet)

controlnet = MultiControlNetModel(controlnets)


v2 = True
base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "ip-adapter-faceid-plusv2_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    controlnet=controlnet,
    feature_extractor=None,
    safety_checker=None
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# load ip-adapter
ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

prompt =  """Generate a highly realistic face swap with the reference face seamlessly integrated into the target image. The result should be photorealistic"""
negative_prompt = "Do not include any visible artifacts, such as plastic-like skin, fish-eye effects, or unrealistic textures"

print(type(target))

print(type(mask))

images = ip_model.generate(
    #image=target,
   # prompt=prompt, negative_prompt=negative_prompt,
     image=target, mask_image=mask, 
   #  face_image=face_image, 
     #   control_images=control_images,
     face_image=target_face_image,
     faceid_embeds=faceid_embeds, shortcut=v2, s_scale=1.0,
     num_samples=1, num_inference_steps=40, seed=20232
)[0]


bg_upsampler = None

arch = 'clean'
channel_multiplier = 2
model_name = 'GFPGANv1.4'
url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'


restorer = GFPGANer(
    model_path="GFPGANv1.4.pth",
    upscale=2,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=bg_upsampler)

# # ------------------------ restore ------------------------
# img_path = "result22.jpg"
# img_name = os.path.basename(img_path)
# print(f'Processing {img_name} ...')
# basename, ext = os.path.splitext(img_name)
# input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

#input_img = np.array(images.convert('RGB'))

#print(input_img.shape)
input_img = cv2.cvtColor(np.array(images), cv2.COLOR_RGB2BGR)
# restore faces and background if necessary
cropped_faces, restored_faces, restored_img = restorer.enhance(
    input_img,

    paste_back=True,
    weight=0.5)

imwrite(restored_img, path_save)