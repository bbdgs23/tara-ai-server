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
            
    def generate_scene(self):
        """Generate random interaction scene and background"""
        scenes = [
            # Quiet Indoor Moments
            {
                "prompt": "person and pet sitting together on a comfortable couch in a cozy living room, warm lighting, natural interaction, soft ambiance, high-end photography",
                "background": "living room",
                "interaction": "sitting together"
            },
            {
                "prompt": "person holding pet in arms with gentle care, warm indoor lighting, intimate moment, soft focus, professional portrait",
                "background": "home",
                "interaction": "holding"
            },
            {
                "prompt": "person cuddling with pet on a window seat, morning sunlight streaming through window, peaceful atmosphere, artistic photography",
                "background": "window seat",
                "interaction": "cuddling"
            },
            {
                "prompt": "person lying on bed with pet curled up next to them, cozy bedroom lighting, intimate bonding moment, lifestyle photography",
                "background": "bedroom",
                "interaction": "resting"
            },
            
            # Active Indoor Play
            {
                "prompt": "person playing with pet using toys in a bright living room, joyful interaction, dynamic movement, candid shot",
                "background": "living room",
                "interaction": "playing"
            },
            {
                "prompt": "person and pet sharing playful moment on floor, natural indoor lighting, genuine laughter, candid photography",
                "background": "home interior",
                "interaction": "floor play"
            },
            
            # Outdoor Activities
            {
                "prompt": "person walking with pet in a beautiful park during golden hour, natural sunlight, trees in background, bokeh effect",
                "background": "park",
                "interaction": "walking"
            },
            {
                "prompt": "person and pet playing fetch in an open field, dynamic action, afternoon sunlight, shallow depth of field",
                "background": "field",
                "interaction": "fetch"
            },
            {
                "prompt": "person and pet at quiet beach during sunset, golden light, waves in background, cinematic shot",
                "background": "beach",
                "interaction": "beach walk"
            },
            
            # Care and Bonding
            {
                "prompt": "person hugging pet with loving expression, gentle embrace, soft indoor lighting, emotional portrait",
                "background": "neutral",
                "interaction": "hugging"
            },
            {
                "prompt": "person petting pet on head with gentle touch, warm lighting, intimate moment, close-up shot",
                "background": "home",
                "interaction": "petting"
            },
            {
                "prompt": "person and pet sharing food moment in kitchen, morning light, caring interaction, lifestyle photography",
                "background": "kitchen",
                "interaction": "feeding"
            },
            
            # Special Settings
            {
                "prompt": "person and pet in cozy cafe, warm ambient lighting, relaxed atmosphere, urban lifestyle shot",
                "background": "cafe",
                "interaction": "cafe time"
            },
            {
                "prompt": "person and pet camping together, evening atmosphere, campfire glow, starry sky, outdoor adventure",
                "background": "campsite",
                "interaction": "camping"
            },
            
            # Additional Poses
            {
                "prompt": "person carrying pet in backpack during hike, outdoor adventure, natural lighting, candid moment",
                "background": "hiking trail",
                "interaction": "backpack carry"
            },
            {
                "prompt": "person and pet sharing hammock moment, garden setting, dappled sunlight, peaceful atmosphere",
                "background": "garden",
                "interaction": "hammock rest"
            },
            {
                "prompt": "person and pet in yoga pose together, serene indoor setting, morning light, wellness lifestyle",
                "background": "yoga space",
                "interaction": "yoga time"
            },
            {
                "prompt": "person giving pet piggyback ride, playful moment, indoor soft lighting, joyful interaction",
                "background": "home",
                "interaction": "piggyback"
            }
        ]
        return np.random.choice(scenes)
            {
                "prompt": "person cuddling with pet on a window seat, soft natural lighting, peaceful atmosphere, morning light through window",
                "background": "window seat area",
                "interaction": "창가에서 교감",
                "description": "창가에서 부드러운 자연광과 함께 반려동물과 포근한 교감을 나누는 장면"
            },
            {
                "prompt": "person lying on bed with pet, reading a book, warm bedroom lighting, intimate bonding moment",
                "background": "bedroom",
                "interaction": "침대에서 휴식",
                "description": "침대에서 책을 읽으며 반려동물과 함께 편안한 시간을 보내는 모습"
            },
            
            # 실내 활동적 상호작용
            {
                "prompt": "person playing with pet using toys in a bright living room, joyful interaction, playful atmosphere",
                "background": "bright living room",
                "interaction": "장난감 놀이",
                "description": "밝은 거실에서 장난감을 가지고 반려동물과 즐겁게 놀아주는 활동적인 장면"
            },
            {
                "prompt": "person training pet in home environment, positive reinforcement, focused interaction, gentle guidance",
                "background": "home training area",
                "interaction": "훈련 시간",
                "description": "가정에서 반려동물과 훈련하며 교감하는 집중된 순간"
            },
            
            # 야외 활동
            {
                "prompt": "person walking with pet in a beautiful park during golden hour, natural sunlight, trees in background, path ahead",
                "background": "park pathway",
                "interaction": "공원 산책",
                "description": "황금빛 햇살이 비치는 공원에서 반려동물과 함께 산책하는 평화로운 모습"
            },
            {
                "prompt": "person and pet playing fetch in an open field, dynamic action, afternoon sunlight, green grass",
                "background": "open field",
                "interaction": "공놀이",
                "description": "넓은 잔디밭에서 반려동물과 공을 던지며 활기차게 놀아주는 역동적인 장면"
            },
            {
                "prompt": "person and pet at a quiet beach, sunset colors, waves in background, peaceful moment",
                "background": "beach",
                "interaction": "해변 산책",
                "description": "석양이 지는 해변에서 반려동물과 함께 여유로운 시간을 보내는 로맨틱한 장면"
            },
            
            # 특별한 순간
            {
                "prompt": "person grooming pet with care and attention, gentle touch, soft indoor lighting, nurturing moment",
                "background": "grooming area",
                "interaction": "그루밍",
                "description": "반려동물을 정성스럽게 손질해주는 애정 어린 교감 장면"
            },
            {
                "prompt": "person and pet sharing a quiet moment in a garden, natural greenery, afternoon light, peaceful setting",
                "background": "garden",
                "interaction": "정원에서 휴식",
                "description": "아름다운 정원에서 반려동물과 조용한 휴식을 즐기는 평화로운 순간"
            },
            {
                "prompt": "person feeding pet in kitchen, morning light through window, caring interaction, homey atmosphere",
                "background": "kitchen",
                "interaction": "급여 시간",
                "description": "부엌에서 반려동물에게 사료를 주는 일상적이면서도 따뜻한 교감 장면"
            },
            
            # 특별한 장소
            {
                "prompt": "person and pet in a cozy cafe setting, indoor plants, warm lighting, relaxed atmosphere",
                "background": "pet-friendly cafe",
                "interaction": "카페에서 휴식",
                "description": "반려동물과 함께 방문한 카페에서 여유로운 시간을 보내는 모습"
            },
            {
                "prompt": "person and pet camping together, campfire light, stars visible, tent in background",
                "background": "campsite",
                "interaction": "캠핑",
                "description": "캠핑장에서 반려동물과 함께 모닥불 앞에서 보내는 특별한 야외 활동"
            }
        ]
        return np.random.choice(scenes)

    def segment_image(self, image):
        """SAM을 사용하여 이미지에서 주요 객체 분할"""
        image_np = np.array(image)
        self.predictor.set_image(image_np)
        
        # 이미지 중앙점 기준으로 예측
        masks = self.predictor.predict(
            point_coords=np.array([[image_np.shape[1]//2, image_np.shape[0]//2]]),
            point_labels=np.array([1]),
            multimask_output=True,
            num_multimask_outputs=3
        )[0]
        
        # 가장 큰 마스크 선택
        mask_idx = np.argmax([mask.sum() for mask in masks])
        return masks[mask_idx]

    def extract_subject(self, image, mask):
        """마스크를 사용하여 배경에서 객체 추출"""
        image_np = np.array(image)
        # 알파 채널 생성
        alpha = (mask * 255).astype(np.uint8)
        # RGBA 이미지 생성
        rgba = np.concatenate([image_np, alpha[..., None]], axis=2)
        return Image.fromarray(rgba)

    def predict(
        self,
        person_image: Path = Input(description="Person image"),
        pet_image: Path = Input(description="Pet image")
    ) -> Path:
        # 이미지 로드 및 전처리
        person = Image.open(person_image).convert("RGB")
        pet = Image.open(pet_image).convert("RGB")
        
        # 이미지 리사이즈 (가로세로 비율 유지)
        target_size = (512, 512)
        person.thumbnail(target_size, Image.LANCZOS)
        pet.thumbnail(target_size, Image.LANCZOS)

        # 사람과 동물 분할
        person_mask = self.segment_image(person)
        pet_mask = self.segment_image(pet)

        # 객체 추출
        person_extracted = self.extract_subject(person, person_mask)
        pet_extracted = self.extract_subject(pet, pet_mask)

        # 임시 파일로 저장
        temp_person = "/tmp/person_extracted.png"
        temp_pet = "/tmp/pet_extracted.png"
        person_extracted.save(temp_person)
        pet_extracted.save(temp_pet)

        try:
            # 랜덤한 장면 생성
            scene = self.generate_scene()
            
            # Flux Fill Pro를 사용하여 이미지 합성
            output = replicate.run(
                "black-forest-labs/flux-fill-pro:3938b53fb07ec2ae4beb3c20d57c7d2c236743516c0f70bc9ef316271727fde6",
                input={
                    "image": open(temp_person, "rb"),
                    "mask_image": open(temp_pet, "rb"),
                    "prompt": scene["prompt"],
                    "negative_prompt": "distorted, unnatural pose, bad composition, unrealistic lighting, artificial looking, deformed, blurry",
                    "num_inference_steps": 30,
                    "guidance_scale": 7.5,
                    "blend_mode": "normal",
                    "mask_feathering": 0.3  # 자연스러운 블렌딩을 위한 페더링
                }
            )

            # 결과 이미지 저장
            output_path = "/tmp/final_composition.png"
            response = requests.get(output[0])
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
            else:
                raise Exception(f"Failed to download image: {response.status_code}")

            return Path(output_path)

        except Exception as e:
            print(f"Error during image synthesis: {str(e)}")
            raise