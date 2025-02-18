FROM r8.im/stability-ai/sdxl@sha256:39ed52f2a78e934b3ba6e2e8e9c0bd39c37c1c6caec87c12f4c3c8cc76d5e859

# 기본 의존성 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 환경 변수 설정
ENV REPLICATE_API_TOKEN=${REPLICATE_API_TOKEN}

# Python 패키지 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# SAM 모델 다운로드
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 애플리케이션 코드 복사
COPY . /app
WORKDIR /app

# Cog 설정
RUN cog init

ENTRYPOINT ["cog", "predict"]