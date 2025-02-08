import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
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
import argparse

def create_face_mask(seg_mask, image_size):
    landmarks = seg_mask['landmark_2d_106']
    landmarks_int = landmarks.astype(np.int32)
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(np.array(landmarks_int))
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    return Image.fromarray(mask)

def _prepare_control_images(tgt) -> list:
    pose_processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    canny_processor = CannyDetector()
    return [
        canny_processor(tgt, threshold=(30, 100)),
        pose_processor(tgt, include_face=True)
    ]

def _align_faces(src_img: np.ndarray, tgt_img: np.ndarray, src_o, tgt_o) -> np.ndarray:
    src_landmarks = src_o['landmark_2d_106']
    tgt_landmarks = tgt_o['landmark_2d_106']
    
    if src_landmarks is None or tgt_landmarks is None:
        return src_img
    
    src_landmarks_int = src_landmarks.astype(np.int32)
    tgt_landmarks_int = tgt_landmarks.astype(np.int32)
    
    transformation_matrix, _ = cv2.estimateAffinePartial2D(
        src_landmarks_int, tgt_landmarks_int, method=cv2.LMEDS
    )
    
    if transformation_matrix is None:
        return src_img
    
    transformed_image = cv2.warpAffine(
        src_img,
        transformation_matrix,
        (src_img.shape[1], src_img.shape[0]),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC
    )
    
    if transformed_image is None or transformed_image.size == 0:
        transformed_image = src_img.copy()
    
    return transformed_image

def main():
    parser = argparse.ArgumentParser(description='Face Swapping with IP-Adapter and GFPGAN')
    parser.add_argument('--source', type=str, required=True, help='Path to source face image')
    parser.add_argument('--target', type=str, required=True, help='Path to target image')
    parser.add_argument('--output', type=str, required=True, help='Path to output image')
    args = parser.parse_args()

    # Validate input paths
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source image not found: {args.source}")
    if not os.path.exists(args.target):
        raise FileNotFoundError(f"Target image not found: {args.target}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Read source and target images
    image = cv2.imread(args.source)
    faces = app.get(image)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)

    target = cv2.imread(args.target)
    target_face = app.get(target)
    target_face_image = face_align.norm_crop(target, landmark=target_face[0].kps, image_size=224)

    mask = create_face_mask(target_face[0], (target.shape[0], target.shape[1]))
    target_pil = Image.open(args.target)

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
    ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

    prompt = "Generate a highly realistic face swap with the reference face seamlessly integrated into the target image. The result should be photorealistic"
    negative_prompt = "Do not include any visible artifacts, such as plastic-like skin, fish-eye effects, or unrealistic textures"

    images = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=target_pil,
        mask_image=mask,
        face_image=target_face_image,
        faceid_embeds=faceid_embeds,
        shortcut=v2,
        s_scale=1.0,
        num_samples=1,
        num_inference_steps=40,
        seed=20232
    )[0]

    restorer = GFPGANer(
        model_path="GFPGANv1.4.pth",
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )

    input_img = cv2.cvtColor(np.array(images), cv2.COLOR_RGB2BGR)
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        paste_back=True,
        weight=0.5
    )

    imwrite(restored_img, args.output)
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()