import torch
from PIL import Image
from cog import BasePredictor, Input, Path
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import replicate
import os
import requests
from io import BytesIO

class Predictor(BasePredictor):
    def setup(self):
        # SAM 모델 초기화 (vit_h 사용)
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(device="cuda")
        self.predictor = SamPredictor(self.sam)
        
        # Replicate API 토큰 설정
        self.api_token = os.environ.get('REPLICATE_API_TOKEN')
        if not self.api_token:
            raise ValueError("REPLICATE_API_TOKEN environment variable is not set")

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
        
        # 이미지 리사이즈
        target_size = (512, 512)
        person = person.resize(target_size, Image.LANCZOS)

        # SAM으로 객체 분할
        person_np = np.array(person)
        
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

        # 마스크 이미지 생성
        mask_image = Image.fromarray((~person_mask[0] * 255).astype(np.uint8))
        
        # 이미지를 임시 파일로 저장 (Replicate API 요구사항)
        temp_image_path = "/tmp/input_image.png"
        temp_mask_path = "/tmp/mask_image.png"
        person.save(temp_image_path)
        mask_image.save(temp_mask_path)

        # 프롬프트 생성
        interaction_prompt = self.generate_interaction_prompt()

        try:
            # Replicate API를 통한 이미지 생성
            output = replicate.run(
                "stability-ai/stable-diffusion-3.5-large:9143d117bc61aa93ac388d31e1fee181e01a5c98ae31f4018c1e755c0714100f",
                input={
                    "prompt": interaction_prompt,
                    "negative_prompt": "low quality, bad composition, unnatural pose, deformed, distorted, disfigured, bad anatomy, blurry",
                    "image": open(temp_image_path, "rb"),
                    "mask": open(temp_mask_path, "rb"),
                    "num_inference_steps": 25,
                    "scheduler": "K_EULER",
                    "guidance_scale": 7.5
                }
            )

            # 결과 이미지 다운로드 및 저장
            output_path = "/tmp/interaction_image.png"
            response = requests.get(output[0])
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
            else:
                raise Exception(f"Failed to download image: {response.status_code}")

            return Path(output_path)

        except Exception as e:
            print(f"Error during image generation: {str(e)}")
            raise