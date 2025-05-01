import os
import shutil

# 경로 설정
sample_dir = '/Volumes/T7'
source_dir = os.path.join(sample_dir, '01.원천데이터')
label_dir  = os.path.join(sample_dir, '02.라벨링데이터')

# 새 디렉토리 경로 설정
data_dir    = os.path.join(sample_dir, 'data')
images_dir  = os.path.join(data_dir, 'images')
labels_dir  = os.path.join(data_dir, 'labels')

# 디렉토리 생성
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# 1) 원천데이터(이미지) 이동
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(images_dir, file)
            shutil.move(src_path, dst_path)
            print(f'이미지 이동: {src_path} -> {dst_path}')

# 2) 라벨링데이터(.json) 이동
for root, dirs, files in os.walk(label_dir):
    for file in files:
        if file.lower().endswith('.json'):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(labels_dir, file)
            shutil.move(src_path, dst_path)
            print(f'라벨 이동: {src_path} -> {dst_path}')


print(f'처리 완료: 이미지는 {images_dir}에, 라벨은 {labels_dir}에 이동되었습니다.')