import torch
from PIL import Image
from cog import BasePredictor, Input, Path
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionXLInpaintPipeline

class Predictor(BasePredictor):
    def setup(self):
        # SDXL 모델 초기화
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16
        ).to("cuda")
        
        # SAM 모델 초기화 (vit_h로 업그레이드)
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(device="cuda")
        self.predictor = SamPredictor(self.sam)

    def generate_interaction_prompt(self):
        interactions = [
            # 산책 및 외출 씬
            "강아지와 함께 공원에서 산책하는 따뜻하고 친근한 scene, 부드러운 자연광, 자연스러운 포즈, soft depth of field, professional photography",
            "고양이를 안고 있는 사람, 따뜻한 미소, 부드러운 배경, 따뜻한 감성의 사진, natural lighting, high resolution",
            "반려동물과 함께 해변을 걷는 모습, 따스한 오후의 햇살, 편안한 복장, high-end photography, soft focus",
            
            # 집 안 상호작용
            "소파에 앉아 강아지 무릎에 앉혀놓고 쓰다듬는 모습, 아늑한 실내, 따뜻한 조명, cozy atmosphere, cinematic lighting",
            "고양이와 함께 창가에 앉아있는 편안한 장면, 부드러운 자연광, golden hour, emotional moment",
            "침대에 누워 반려동물과 함께 휴식을 취하는 아늑한 scene, soft lighting, intimate atmosphere",
            
            # 놀이 및 활동
            "공원에서 공을 던지며 강아지와 놀아주는 사람, 활기차고 생동감 넘치는 scene, dynamic composition, shallow depth of field",
            "반려동물을 들어올리며 즐겁게 웃고 있는 모습, 밝고 따뜻한 배경, joyful moment, professional portrait",
            
            # 특별한 순간
            "강아지 머리를 쓰다듬으며 포옹하는 친근한 모습, 따뜻한 색감, emotional connection, artistic photography",
            "반려동물과 서로를 바라보며 깊은 교감을 나누는 모멘트, 부드러운 bokeh 효과, intimate portrait, high-end photography"
        ]
        return np.random.choice(interactions)

    def predict(
        self,
        person_image: Path = Input(description="Person image"),
        pet_image: Path = Input(description="Pet image")
    ) -> Path:
        # 이미지 전처리
        person = Image.open(person_image).convert("RGB")
        pet = Image.open(pet_image).convert("RGB")
        
        # 이미지 리사이즈
        target_size = (512, 512)
        person = person.resize(target_size, Image.LANCZOS)
        pet = pet.resize(target_size, Image.LANCZOS)

        # SAM으로 객체 분할
        person_np = np.array(person)
        pet_np = np.array(pet)
        
        self.predictor.set_image(person_np)
        person_mask = self.predictor.predict(
            point_coords=np.array([[person_np.shape[1]//2, person_np.shape[0]//2]]),
            point_labels=np.array([1]),
            multimask_output=True,
            num_multimask_outputs=3
        )[0]

        # 가장 큰 마스크 선택
        person_mask_idx = np.argmax([mask.sum() for mask in person_mask])
        person_mask = person_mask[person_mask_idx:person_mask_idx+1]

        # 마스킹된 이미지 생성
        person_masked = person_np.copy()
        person_masked[~person_mask[0]] = [0, 0, 0]

        # 상호작용 프롬프트 생성
        interaction_prompt = self.generate_interaction_prompt()

        # SDXL로 최종 이미지 생성
        result = self.pipe(
            prompt=interaction_prompt,
            negative_prompt="low quality, bad composition, unnatural pose, deformed, distorted, disfigured, bad anatomy, blurry",
            image=Image.fromarray(person_masked),
            mask_image=Image.fromarray((~person_mask[0] * 255).astype(np.uint8)),
            num_inference_steps=25,  # 스텝 수 증가
            strength=0.8,
            guidance_scale=7.5
        ).images[0]

        # 최종 이미지 저장
        output_path = "/tmp/interaction_image.png"
        result.save(output_path)
        
        return Path(output_path)