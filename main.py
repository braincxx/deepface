
import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
#MULTI (work)
from PIL import Image

import numpy as np
import mediapipe as mp
from controlnet_aux import OpenposeDetector, CannyDetector

def _generate_face_mask( image: Image.Image) -> Image.Image:
        """Генерация маски лица с мягкими краями"""
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        image_np = np.array(image)
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        
        results = mp_face_mesh.process(image_np)
       
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
           
            face_oval = mp.solutions.face_mesh.FACEMESH_FACE_OVAL
            points = [(int(landmarks.landmark[i].x * image.width), 
                      int(landmarks.landmark[i].y * image.height)) 
                     for i, _ in face_oval]
            
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.GaussianBlur(mask, (51, 51), 0)
            
        return Image.fromarray(mask)




def create_face_mask(seg_mask, image_size):
    """
    Creates a binary mask from face landmarks for inpainting.
    
    Args:
        seg_mask (dict): Dictionary containing face detection results.
        image_size (tuple): Tuple (height, width) of the image.
        
    Returns:
        numpy.ndarray: Binary mask where the face region is white.
    """
    # Extract the 2D landmarks
   
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
    return mask

# def create_face_mask(seg_mask, image_size):
#     """
#     Creates a binary mask from a bounding box for inpainting.

#     Args:
#         seg_mask (dict): Dictionary containing face detection results.
#         image_size (tuple): Tuple (height, width) of the image.

#     Returns:
#         numpy.ndarray: Binary mask where the bbox region is white.
#     """
#     # Extract the bounding box coordinates
#     bbox = seg_mask['bbox']

#     # Create a blank mask of the same size as the image
#     height, width = image_size
#     mask = np.zeros((height, width), dtype=np.uint8)

#     # Draw the filled rectangle of the bbox
#     x1, y1, x2, y2 = bbox.astype(np.int32)
#     cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

#     return mask


app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))





image = cv2.imread("1.jpg")
faces = app.get(image)

faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face

path_target= "target_1.jpg"
target = cv2.imread(path_target)
target_face = app.get(target)
target_face_image = face_align.norm_crop(target, landmark=target_face[0].kps, image_size=224) 


def _prepare_control_images(tgt) -> list:
    """Подготовка контрольных изображений"""
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
     #Calculate the transformation matrix
    transformation_matrix, _ = cv2.estimateAffinePartial2D(
        src_landmarks_int, tgt_landmarks_int, method=cv2.LMEDS
    )

    if transformation_matrix is None:
        return src_img

    # Apply the transformation to the source image
    return cv2.warpAffine(
        src_img,
        transformation_matrix,
        (src_img.shape[1], src_img.shape[0]),
        flags=cv2.INTER_CUBIC
    )

control_images = _prepare_control_images(target) + [Image.fromarray(_align_faces(face_image, target_face_image, faces[0], target_face[0]))]
Image.fromarray(_align_faces(face_image, target_face_image, faces[0], target_face[0])).save("example_transform.jpeg")
#print("SEG MASK", target_face[0])
# For segmented masks (FaceSwap Lab's "improved mask" option)


mask = create_face_mask(target_face[0], (target.shape[0], target.shape[1]))

#mask = _generate_face_mask(Image.open("target_1.jpg"))
#mask.save("face-mask.jpg")
#cv2.imwrite('face_mask.png', mask)
Image.fromarray(mask).save("face-mask.jpg")
Image.fromarray(target_face_image).save("target-face.jpg")

target = Image.open(path_target)
mask = Image.fromarray(mask)


import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, DDIMScheduler, ControlNetModel, AutoencoderKL, StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy
from PIL import Image

from diffusers.pipelines.controlnet import MultiControlNetModel
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

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
#pipe = StableDiffusionInpaintPipeline.from_pretrained(

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


images = ip_model.generate(
    #image=target,
     image=target, mask_image=mask, 
   #  face_image=face_image, 
     #   control_images=control_images,
     face_image=target_face_image,
     faceid_embeds=faceid_embeds, shortcut=v2, s_scale=1.0,
     num_samples=1, num_inference_steps=35, seed=20232
)[0]

images.save("result22.jpg")
