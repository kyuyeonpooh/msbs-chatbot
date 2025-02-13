# Mitsubishi Chatbot Project
미쓰비시 전시회용 챗봇 프로젝트입니다.

## How to install
1. Python >= 3.11 을 만족하는지 확인해주세요.

2. Package manager인 uv를 설치합니다.
```shell
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```
3. `make dev`를 통해 설치합니다.
```shell
$ make dev
```

## How to run
1. OpenAI API Key가 환경 변수로 등록되어 있는지 확인해주세요.
```shell
OPENAI_API_KEY=...
```
2. Streamlit 대시보드를 실행합니다.
```shell
$ streamlit run main.py
```
