import numpy as np
from imageio import imwrite
from imageio import imwrite

# 데이터 불러오기

data = np.loadtxt(r"input path.txt",skiprows=7)

x = data[:, 0]
y = data[:, 1]
z = data[:, 2]


# 고유한 x, y 길이로 height map 복원

x_unique = np.unique(x)
y_unique = np.unique(y)

nx = len(x_unique)
ny = len(y_unique)

height_map = np.zeros((ny, nx))

x_index = {v: i for i, v in enumerate(x_unique)}
y_index = {v: i for i, v in enumerate(y_unique)}

for xi, yi, zi in zip(x, y, z):
    ix = x_index[xi]
    iy = y_index[yi]
    height_map[iy, ix] = zi


# top / bottom height 추출

h_bottom = np.min(height_map)
h_top = np.max(height_map)

print("Top height:", h_top)
print("Bottom height:", h_bottom)


# 원하는 높이에서 threshold 설정

# (A) Threshold_ratio
threshold_ratio = 0.5  # ex) 0.5 = 50% height, 0.3 = 30%, 0.7 = 70%
h_thr_ratio = h_bottom + threshold_ratio * (h_top - h_bottom)

# (B) Absolute height (nm)
use_absolute_height = False   # True -> absolute height
absolute_height = -10         

if use_absolute_height:
    h_thr = absolute_height
    print(f"Using absolute height threshold = {absolute_height} nm")
else:
    h_thr = h_thr_ratio
    print(f"Using ratio threshold = {threshold_ratio * 100:.1f}% → height {h_thr}")


# binary 이미지 생성

binary_mask = (height_map >= h_thr).astype(np.uint8)
binary_img = binary_mask * 255


# binary 이미지 저장

output_name = f"test_threshold_{threshold_ratio:.2f}.png"
imwrite(output_name, binary_img)

print(f"Saved '{output_name}'")


# binary img에서 좌표 추출

ny, nx = binary_mask.shape

# nm/pixel 계산 (AFM scan 조건)
scan_size_x_um = 2.5                 # 여기에 실제 X scan size(µm) 
scan_size_x_nm = scan_size_x_um * 1000.0
dx_nm = scan_size_x_nm / nx

print("nx =", nx)
print("dx_nm (nm/pixel) =", dx_nm)


# 각 row에서 line CD 계산

cd_nm_all = []

for iy in range(ny):
    row = binary_mask[iy, :].astype(int)

    # edge 검출
    edges = np.where(np.diff(row) != 0)[0]

    # 검은색 run 목록 생성
    runs = []
    current_value = row[0]
    start = 0

    for e in edges:
        end = e
        if current_value == 0:    # 검은색이 line인 경우
            runs.append((start, end))
        start = e + 1
        current_value = 1 - current_value

    # 마지막 구간 처리
    end = nx - 1
    if current_value == 0:
        runs.append((start, end))

    
    # 양쪽 run 제거 
    
    if len(runs) > 2:
        runs = runs[1:-1]   # 왼쪽 run, 오른쪽 run 제외
    else:
        continue            # run이 너무 적으면 건너뛰기

    
    # 남은 run들에 대해 CD 계산
    
    for (left, right) in runs:
        cd_px = right - left + 1
        cd_nm = cd_px * dx_nm
        cd_nm_all.append(cd_nm)

cd_nm_all = np.array(cd_nm_all)

print("라인 개수 (총 측정된 CD 수):", len(cd_nm_all))
print("평균 CD (nm):", np.mean(cd_nm_all))
print("CD 3σ (nm):", 3 * np.std(cd_nm_all))
print("CD 최소/최대 (nm):", np.min(cd_nm_all), np.max(cd_nm_all))
