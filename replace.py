import random
import shutil
from pathlib import Path

def split_dataset(output_dir: str, train_ratio: float = 0.8):
    """
    output_dir/
      images/
      labels/
    구조를
    output_dir/
      train/images/
      train/labels/
      val/images/
      val/labels/
    로 복사(또는 이동)합니다.
    """
    out = Path(output_dir)
    img_dir = out / 'images'
    lbl_dir = out / 'labels'

    # 모든 이미지 파일 리스트
    image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    random.shuffle(image_files)

    total = len(image_files)
    train_count = int(total * train_ratio)

    splits = {
        'train': image_files[:train_count],
        'val':   image_files[train_count:],
    }

    for split_name, files in splits.items():
        tgt_img_dir = out / split_name / 'images'
        tgt_lbl_dir = out / split_name / 'labels'
        tgt_img_dir.mkdir(parents=True, exist_ok=True)
        tgt_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            # 이미지 복사
            shutil.move(img_path, tgt_img_dir / img_path.name)
            # 대응 라벨 복사
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy(lbl_path, tgt_lbl_dir / lbl_path.name)

    print(f"총 {total}개의 샘플 중 {train_count}개를 train, {total-train_count}개를 val로 분할했습니다.")

if __name__ == '__main__':
    # --output-dir 인자값을 넣어주세요.
    split_dataset(output_dir='/Volumes/T7/result', train_ratio=0.8)