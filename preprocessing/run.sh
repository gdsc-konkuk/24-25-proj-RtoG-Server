#!/bin/bash
# 이미지 전처리 도커 컨테이너 빌드 및 실행 스크립트

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 기본 매개변수 설정
INPUT_DIR="../data"
OUTPUT_DIR="../processed"

# 사용법 함수
function print_usage {
    echo "사용법: $0 [옵션]"
    echo "옵션:"
    echo "  --input-dir DIR     입력 디렉토리 경로 (기본값: ../data)"
    echo "  --output-dir DIR    출력 디렉토리 경로 (기본값: ../processed)"
    echo "  --help              도움말 출력"
}

# 명령행 인수 처리
while [ "$#" -gt 0 ]; do
    case "$1" in
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --help) print_usage; exit 0 ;;
        *) echo "알 수 없는 옵션: $1"; print_usage; exit 1 ;;
    esac
done

echo -e "${GREEN}=== RtoG 이미지 전처리 도커 스크립트 ===${NC}"
echo "입력 디렉토리: ${INPUT_DIR}"
echo "출력 디렉토리: ${OUTPUT_DIR}"

# 절대 경로 변환
INPUT_DIR_ABS=$(realpath "${INPUT_DIR}")
OUTPUT_DIR_ABS=$(realpath "${OUTPUT_DIR}")

# 현재 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 도커 이미지 빌드
echo -e "${YELLOW}도커 이미지 빌드 중...${NC}"
docker build -t rtog-preprocessing .

# 입출력 디렉토리 생성
mkdir -p "${INPUT_DIR_ABS}/images" "${INPUT_DIR_ABS}/labels" "${OUTPUT_DIR_ABS}/images" "${OUTPUT_DIR_ABS}/labels"

# 도커 실행
echo -e "${YELLOW}이미지 전처리 컨테이너 실행 중...${NC}"
docker run --rm \
  -v "${INPUT_DIR_ABS}:/app/data" \
  -v "${OUTPUT_DIR_ABS}:/app/processed" \
  rtog-preprocessing --input-dir /app/data --output-dir /app/processed

echo -e "${GREEN}처리 완료!${NC}"
echo "결과물은 ${OUTPUT_DIR} 디렉토리에 저장되었습니다." 