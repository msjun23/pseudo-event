from PIL import Image
import os

# 이미지 파일 경로 패턴 설정
image_dir = 'save/event_representation/zurich_city_05_a'
image_prefix = '000'
start_index = 580
end_index = 632
file_extension = '.png'

# 이미지 파일 리스트 생성
image_files = [
    os.path.join(image_dir, f"{image_prefix}{str(i).zfill(3)}{file_extension}")
    for i in range(start_index, end_index + 1, 2)  # 2씩 증가시키면서 파일명 생성
]

# 이미지를 읽고 PIL 이미지 객체로 변환
images = [Image.open(image) for image in image_files]

# GIF 저장
output_path = 'assets/output.gif'
images[0].save(output_path, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

print(f"GIF 파일이 {output_path}에 저장되었습니다.")