# Xử lý NAVSIM mini cho METEOR DataLoader

Tài liệu này mô tả đúng pipeline được triển khai trong
`bevlane/ingest_navsim.py` và phần hỗ trợ `image_root` trong
`bevlane/dataset.py`. Mục tiêu là dùng trực tiếp OpenScene/NAVSIM mini đã tải ở
`D:\navsim_workspace\dataset`, không sao chép lại hơn 151 GiB sensor blobs.

## 1. Kết quả hiện có trên máy

Pipeline nhẹ đã được chạy cho toàn bộ mini:

```text
D:\navsim_workspace\meteor_mini
├── scenes.txt
├── <log-name-1>
│   ├── manifest.json
│   ├── manifest.before_gt_promotion.json  # bản sao trước khi đổi gt
│   ├── manifest.before_agent_traj.json    # bản sao trước khi thêm trajectory
│   ├── ego_motion.npz
│   ├── gt_map\000000.png ...          # gt mặc định hiện trỏ tới các file này
│   ├── bev_box\000000.npz ...
│   └── agent_traj\000000.npz ...
└── <log-name-64>\...
```

Kết quả đã kiểm tra:

- 64 recording/log;
- 51.867 frame ở 2 Hz;
- 51.457 frame có đủ quỹ đạo tương lai 3 giây;
- ảnh không được copy, mà được đọc lazy từ raw `sensor_blobs\mini`;
- 51.867 map raster PNG, đủ file trong manifest của 64/64 recording;
- `gt` và `gt_map` cùng trỏ tới map raster ở tất cả 51.867 frame;
- 64 file `gt/ignore.png` và 64 thư mục `gt` rỗng đã được xoá sau khi kiểm tra;
- 51.867 agent-trajectory target đã được tạo từ `track_tokens`, kèm backup của
  64 manifest trước khi thêm target;
- batch thật đã load thành công qua `torch.utils.data.DataLoader` với
  `gt_key="gt_map"` ở cả bốn khu vực (sau promotion, mặc định `gt` cũng vậy).

Pipeline này dùng ngay được cho input camera, calibration, 3D detection, ego
trajectory, raw driving command và agent trajectory. Gói nuPlan maps được lưu riêng ở
`D:\navsim_workspace\dataset\maps`; `bevlane/rasterize_navsim_map.py` tạo
map-derived BEV target vào `gt_map/`. Bước
`bevlane/promote_navsim_map_gt.py` chuyển trường `gt` mặc định sang các target
đã xác minh và xoá placeholder.
Depth/LiDAR BEV vẫn chưa materialize cho toàn bộ 51.867 frame.

## 2. Dữ liệu raw NAVSIM mini có gì?

### 2.1 Cấu trúc thư mục

```text
D:\navsim_workspace\dataset
├── navsim_logs\mini
│   ├── <log-name>.pkl
│   └── ...                         # 64 pickle logs
└── sensor_blobs\mini
    ├── <log-name>
    │   ├── CAM_F0\*.jpg
    │   ├── CAM_L0\*.jpg
    │   ├── CAM_L1\*.jpg
    │   ├── CAM_L2\*.jpg
    │   ├── CAM_R0\*.jpg
    │   ├── CAM_R1\*.jpg
    │   ├── CAM_R2\*.jpg
    │   ├── CAM_B0\*.jpg
    │   └── MergedPointCloud\*.pcd
    └── ...
```

### 2.2 File log `.pkl`

Mỗi `.pkl` là Python pickle chứa `list[dict]`; mỗi dictionary là một frame.
Các trường pipeline đang dùng:

| Trường | Kiểu/shape | Ý nghĩa |
|---|---:|---|
| `token` | `str` | ID duy nhất của frame |
| `timestamp` | `int` microsecond | Thời gian frame |
| `frame_idx` | `int` | Chỉ số trong recording |
| `cams` | `dict` 8 camera | Path ảnh, intrinsic, distortion, extrinsic |
| `lidar_path` | `str` | Path tương đối tới merged point cloud |
| `ego2global_translation` | `[3]` | Vị trí ego trong hệ global |
| `ego2global_rotation` | quaternion `[w,x,y,z]` | Hướng ego trong global |
| `ego_dynamic_state` | `[vx,vy,ax,ay]` | Vận tốc và gia tốc ego |
| `driving_command` | one-hot `[4]` | left/straight/right/unknown của NAVSIM |
| `anns.gt_boxes` | `[N,7]` | `[x,y,z,length,width,height,yaw]` trong local LiDAR |
| `anns.gt_names` | `[N]` | `vehicle`, `pedestrian`, `bicycle`, ... |

Log còn có route, traffic light, occupancy/flow path và token tracking. Bản
converter hiện tại không giả định rằng các file occupancy được tải kèm; nó chỉ
dùng các trường đã xác minh tồn tại trong mini.

### 2.3 Camera JPEG

Mỗi frame có tám ảnh JPEG. Ví dụ đã kiểm tra có kích thước gốc 1920×1080.
Camera dictionary chứa:

- `data_path`: path tương đối dưới `sensor_blobs\mini`;
- `cam_intrinsic`: ma trận pinhole `K [3,3]`;
- `sensor2lidar_rotation [3,3]`;
- `sensor2lidar_translation [3]`;
- `distortion [5]`.

NAVSIM quy ước:

```text
p_lidar = R_sensor_to_lidar · p_camera + t_sensor_to_lidar
```

METEOR manifest lưu chính ma trận camera → ego/LiDAR này dưới tên
`T_ego_cam`. Khi load, `BevLaneDataset` đảo ma trận để thu được
`T_cam_ego`, dùng cho phép chiếu điểm BEV vào ảnh.

### 2.4 Point cloud `.pcd`

Mini dùng PCD v0.7, `DATA binary`, một record 15 byte:

| Field | Kiểu |
|---|---|
| `x`, `y`, `z` | little-endian `float32` |
| `intensity` | `uint8` |
| `lidar_info` | `uint8` |
| `ring` | `uint8` |

`bevlane/ingest_navsim.py::pcd_xyz` parse header thay vì hard-code offset, sau
đó lấy `xyz [N,3]` trong merged-LiDAR/local ego frame.

## 3. Ánh xạ camera NAVSIM → slot METEOR

METEOR có tám slot lịch sử. NAVSIM có tám camera surround-view nhưng tên khác:

| Slot METEOR | Camera NAVSIM | Ghi chú |
|---|---|---|
| `CAM_FRONT_WIDE` | `CAM_F0` | trước |
| `CAM_FRONT_LEFT` | `CAM_L0` | trước-trái |
| `CAM_FRONT_RIGHT` | `CAM_R0` | trước-phải |
| `CAM_BACK_WIDE` | `CAM_B0` | sau |
| `CAM_BACK_LEFT` | `CAM_L2` | sau-trái |
| `CAM_BACK_RIGHT` | `CAM_R2` | sau-phải |
| `CAM_FRONT_NARROW` | `CAM_L1` | camera bên trái vào slot narrow thứ nhất |
| `CAM_BACK_NARROW` | `CAM_R1` | camera bên phải vào slot narrow thứ hai |

Hai slot cuối không có cùng hướng nhìn với tên lịch sử của rig METEOR. Đây là
ánh xạ zero-shot giống adapter `navsim_meteor`; nó giữ đủ tám ảnh nhưng không
biến hai rig thành cùng một calibration vật lý.

## 4. Pipeline xử lý

### 4.1 Không copy ảnh

`manifest.json` có trường:

```json
{
  "image_root": "D:\\navsim_workspace\\dataset\\sensor_blobs\\mini",
  "img_hw": [432, 768]
}
```

Mỗi `frames[i].imgs` chỉ chứa path tương đối tới ảnh raw. `BevLaneDataset`:

1. resolve ảnh bằng `image_root`;
2. đọc BGR bằng OpenCV;
3. resize về 768×432;
4. đổi BGR → RGB;
5. đổi `[0,255]` thành `[0,1]`;
6. chuẩn hóa ImageNet:
   `image = (image - [0.485,0.456,0.406]) / [0.229,0.224,0.225]`;
7. chuyển thành `[camera, channel, height, width]`.

Intrinsic cũng được scale đồng bộ:

```text
sx = 768 / source_width
sy = 432 / source_height
K'[0,:] = sx · K[0,:]
K'[1,:] = sy · K[1,:]
```

Ví dụ camera trước của log đầu: `fx=fy=1545` ở ảnh 1920×1080 trở thành
`fx=fy=618`, `cx=384`, `cy=224` ở 768×432.

### 4.2 Ego trajectory

NAVSIM mini là 2 Hz, tức 0,5 giây/frame. Converter tạo `ego_motion.npz`:

| Array | Shape | Nội dung |
|---|---:|---|
| `wp` | `[F,6,2]` | vị trí tương lai +0,5…+3,0 s trong ego hiện tại |
| `v0` | `[F]` | `sqrt(vx²+vy²)` m/s |
| `acc` | `[F]` | gia tốc dọc từ `ego_dynamic_state[2]` |
| `steer` | `[F]` | `atan(wheelbase · yaw_rate / speed)` |
| `brake` | `[F]` | 1 nếu `acc < -0,5 m/s²` |
| `valid` | `[F]` | 1 nếu còn đủ 3 giây tương lai |
| `pose` | `[F,3]` | global `[x,y,yaw]`, dùng cho temporal warp |
| `stamp` | `[F]` | timestamp giây |

Tọa độ waypoint:

```text
dx = x_future - x_now
dy = y_future - y_now
x_ego =  cos(yaw)·dx + sin(yaw)·dy     # forward
y_ego = -sin(yaw)·dx + cos(yaw)·dy     # left
```

Nếu dùng `--stride > 1`, converter vẫn nội suy trên toàn bộ track raw tại đúng
mốc 0,5 giây; nó không nhầm một frame đã stride thành 0,5 giây.

### 4.3 3D boxes

NAVSIM box `[x,y,z,length,width,height,yaw]` đã ở local LiDAR. Converter:

- `vehicle` → METEOR class 1;
- `pedestrian`, `bicycle` → METEOR class 2 (VRU);
- `traffic_cone`, `barrier`, `generic_object`, `czone_sign` không được đưa vào
  head Vehicle/VRU;
- lọc ngoài vùng BEV ±80 m dọc, ±50 m ngang (có margin 5 m);
- sort gần → xa và giữ tối đa 64 box;
- lưu `[class,x,y,length,width,yaw]` vào `bev_box/*.npz`.

`--write-box-raster` tạo thêm footprint `800×500 @ 0,2 m`, nhưng head detection
tham số chỉ cần file NPZ.

### 4.4 LiDAR BEV tùy chọn

`--with-lidar-bev` tạo tensor `[4,400,250] @ 0,4 m`:

1. `log(1 + số điểm)`;
2. max-z;
3. mean-z;
4. occupancy 0/1.

Quy đổi pixel:

```text
row = (80 - x_forward) / 0.4
col = (50 - y_left) / 0.4
```

File được lưu float16 với key `lb`, còn DataLoader trả float32.

### 4.5 Sparse depth tùy chọn

`--with-depth` chiếu merged point cloud vào từng camera:

```text
p_camera = inverse(T_ego_cam) · p_ego
u = fx·X/Z + cx
v = fy·Y/Z + cy
```

Chỉ giữ `0,5 < Z < 79 m`; nếu nhiều điểm rơi vào cùng cell stride-4 thì lấy
Z gần nhất. Output là `[8,108,192]`, tách theo contract cũ:

- `depth4`: 6 camera đầu;
- `depth4n`: 2 slot cuối.

Đây là sparse camera-z depth, không phải Euclidean range và không phải dense
depth đã được panoptic densification.

### 4.6 BEV segmentation từ nuPlan map

Converter nền ban đầu tạo `gt/ignore.png` (800×500, toàn 255) để không gán
background giả *trước khi có map*. Bước `bevlane/rasterize_navsim_map.py` đọc
gói maps riêng và tạo `gt_map/<frame>.png` cùng kích thước. Sau khi kiểm tra đủ
mọi target, `bevlane/promote_navsim_map_gt.py` đã đổi trường `gt` trong
manifest sang chính `gt_map` và xoá ảnh trắng; không còn ảnh giữ chỗ trong
output hiện tại. Trong METEOR, 255 là `ignore_index` cho các ô chưa được map
phủ, không phải background.

Quy trình của bước map:

1. Đọc `map_location` trong log để chọn GeoPackage thành phố. Metadata
   `projectedCoordSystem` trong `map.gpkg` cung cấp UTM EPSG tương ứng.
2. Parse các vector WGS84 trong GeoPackage và chuyển sang UTM của ego pose.
   Không giả định mọi thành phố dùng cùng một vùng UTM.
3. Lấy hình học nằm trong cửa sổ BEV quanh ego và biến đổi global → local:

   ```text
   dx = map_x - ego_x; dy = map_y - ego_y
   x_forward = cos(yaw)·dx + sin(yaw)·dy
   y_left    = -sin(yaw)·dx + cos(yaw)·dy
   row       = (80 - x_forward) / 0.2
   col       = (50 - y_left) / 0.2
   ```

4. Vẽ polygon và line theo thứ tự lớp; các ô không có hình học luôn là 255,
   **không** phải class 0/background.
5. Chỉ đưa key `gt_map` vào manifest sau khi mọi frame của recording được tạo
   và tỷ lệ ego-center nằm trong map đạt ngưỡng kiểm tra (mặc định 75%).

| METEOR class | Nguồn vector nuPlan | Mức độ khớp nghĩa |
|---:|---|---|
| 1 road | `lanes_polygons`, lane-connector polygons, `intersections`, `generic_drivable_areas` | drivable geometry của map |
| 2 sidewalk | `walkways` | polygon lối đi bộ |
| 3 crosswalk | `crosswalks` | polygon qua đường |
| 4 laneline | boundary được ≥2 lane polygon tham chiếu | **proxy hình học giữa lane**, không chứng minh có vạch sơn |
| 6 road_edge | boundary chỉ được 1 lane polygon tham chiếu | **proxy mép lane**, có thể khác mép đường thực |
| 255 ignore | ngoài các vector trên | không có supervision |

Các class 5 stopline, 7 marking, 8 parking không được suy diễn từ map khi
chưa có ánh xạ taxonomy đã kiểm chứng. `roadblock_ids` mô tả *route*, không
cần để tạo raster của **toàn bộ** lane gần ego; nếu cần route-only target thì
phải lọc hình học theo các ID này trong một bước riêng.

Đây là **map-derived supervision**, không phải nhãn vạch sơn quan sát từ ảnh.
Không nên so IoU của class 4 với painted lane GT như thể hai khái niệm giống
nhau. Giá trị phù hợp hơn là huấn luyện/prior hình học và kiểm tra alignment.

## 5. Cách chạy

### Script chuẩn end-to-end (khuyến nghị)

Sửa duy nhất `DATASET_ROOT` ở đầu `scripts/prepare_navsim_mini.py`:

```python
DATASET_ROOT = Path(r"E:\navsim_workspace\dataset")
```

`OUTPUT_ROOT` mặc định được suy ra thành thư mục `meteor_mini` nằm cạnh
dataset raw. Sau đó chạy trực tiếp, không cần khai báo dataset trong
terminal:

```powershell
py -3.12 scripts\prepare_navsim_mini.py
```

Script chạy theo thứ tự: preflight raw data → ingest → map GT → promotion
→ agent trajectory → kiểm tra mọi reference → DataLoader smoke. Pipeline có
thể chạy lại để resume; stage đã hoàn chỉnh sẽ skip. Các option
CLI cũ vẫn có thể dùng khi cần ghi đè cấu hình trong file. Smoke
test mặc định dùng `num_workers=0`, batch 1 vì mỗi sample tám camera
float32 chiếm khoảng 30,4 MiB; nhiều Windows worker có thể làm hết RAM.

Xem trước các subprocess mà không xử lý:

```powershell
py -3.12 -m scripts.prepare_navsim_mini `
  --dry-run
```

Đổi smoke 16 frame thành full I/O 51.867 frame bằng `--full-io`. Recipe chuẩn
này cố ý chưa materialize LiDAR BEV/sparse depth; hai target đó là một stage
tùy chọn riêng và không được bật ngầm.

Nếu máy không có nuPlan maps, chạy chế độ không có map:

```powershell
py -3.12 -m scripts.prepare_navsim_mini `
  --data-root "E:\navsim_workspace\dataset" `
  --out "E:\navsim_workspace\meteor_mini_no_map" `
  --skip-map-gt `
  --workers 4 `
  --io-workers 6 `
  --io-batch-size 2 `
  --io-prefetch-factor 1
```

Chế độ này không yêu cầu `<data-root>/maps`, bỏ qua rasterize/promotion và giữ
`gt/ignore.png` toàn 255 vì DataLoader vẫn cần một tensor `gt [800,500]`. Đây
không phải lane GT: mọi pixel đều là `ignore_index`, vì vậy chỉ camera,
calibration, box, ego, raw command và agent trajectory có supervision. Nếu dùng
trainer, đặt `--seg-w 0` và không báo cáo lane/road IoU từ output này. Sau khi
có maps, chạy lại cùng output **không có** `--skip-map-gt`; pipeline sẽ bổ sung
`gt_map`, promotion và xóa placeholder an toàn.

### 5.1 Pipeline nhẹ toàn bộ mini

Lệnh dưới đây đã chạy thành công trên máy; chạy lại sẽ skip scene đã có
`manifest.json`:

```powershell
py -3.12 -m bevlane.ingest_navsim `
  --data-root "D:\navsim_workspace\dataset" `
  --out "D:\navsim_workspace\meteor_mini" `
  --split mini `
  --workers 4 `
  --verify
```

### 5.2 Smoke test đầy đủ các tensor tùy chọn

```powershell
py -3.12 -m bevlane.ingest_navsim `
  --data-root "D:\navsim_workspace\dataset" `
  --out "out\navsim_smoke" `
  --split mini `
  --max-logs 1 `
  --max-frames 2 `
  --with-lidar-bev `
  --with-depth `
  --write-box-raster `
  --verify
```

Kết quả đã đo:

```text
imgs       (2, 8, 3, 432, 768)
K          (2, 8, 3, 3)
T_cam_ego  (2, 8, 4, 4)
gt         (2, 800, 500)
depth      (2, 8, 108, 192)
boxes      (2, 64, 6)
box_count  (2,)
ego        (2, 17)
lidar_bev  (2, 4, 400, 250)
```

Ổ D còn khoảng 29 GiB sau khi thêm maps. Không nên chạy full
`--with-depth --with-lidar-bev` cùng lúc trước khi đo kích thước trên một vài
log hoặc giải phóng thêm dung lượng.

### 5.3 Tạo BEV map target

Maps dùng bởi NAVSIM v2.2 đã được đặt đúng cấu trúc:

```text
D:\navsim_workspace\dataset\maps\nuplan-maps-v1.0.json
D:\navsim_workspace\dataset\maps\<city>\<version>\map.gpkg
```

Archive `nuplan-maps-v1.1.zip` được tải từ mirror Hugging Face
`pengxiang/nuplan_maps`, dung lượng 970.997.691 byte và SHA-256
`444860429f9a3bcf89a6459d683fa82eb9219aa259d8d9b8cdefdb37f0b56b05`
đúng với LFS metadata của mirror. Đây là cùng tên/dung lượng gói mà script
NAVSIM v2.2 yêu cầu từ Motional S3; bốn `map.gpkg` sau giải nén đều qua
`PRAGMA integrity_check=ok`. ZIP được giữ ở
`D:\navsim_workspace\maps_hf_staging\nuplan-maps-v1.1.zip` để phục hồi.

Lần chạy đầy đủ trên máy tạo 51.867 PNG (khoảng 0,839 GiB), với 64/64
recording vượt ngưỡng ego-center coverage; lớp `1,2,3,4,6,255` được đọc qua
DataLoader. Ego-center coverage 100% là *sanity check tọa độ*, không phải thước
đo độ chính xác từng pixel của map target.

Để tái chạy bước map (cần `shapely>=2`, `pyproj`, `opencv-python`, `numpy`;
`pyproj 3.8.0` đã được cài vào Python 3.12 trên máy này):

```powershell
py -3.12 -m bevlane.rasterize_navsim_map `
  --data-root "D:\navsim_workspace\dataset" `
  --out "D:\navsim_workspace\meteor_mini" `
  --split mini `
  --workers 4
```

Script bỏ qua recording đã rasterize đầy đủ. `--force` chỉ dùng khi chủ động
muốn tái tạo map target. `--max-frames` là kiểm tra nhanh và **không** đánh dấu
manifest là hoàn tất khi mới xử lý một phần recording.

Sau khi toàn bộ map target đã được kiểm chứng, đưa nó vào `gt` mặc định và xoá
placeholder bằng thao tác có preflight. Lệnh không có `--apply` chỉ kiểm tra:

```powershell
py -3.12 -m bevlane.promote_navsim_map_gt `
  --root "D:\navsim_workspace\meteor_mini"

py -3.12 -m bevlane.promote_navsim_map_gt `
  --root "D:\navsim_workspace\meteor_mini" `
  --apply
```

Lệnh `--apply` đã chạy trên 64 recording. Nó tạo
`manifest.before_gt_promotion.json` trong từng scene, đổi `gt` của mọi frame
sang `gt_map`, xác minh lại manifest rồi mới xoá đúng ảnh `gt/ignore.png` đã
kiểm tra là toàn 255. Raw sensors, maps và `gt_map/*.png` không bị xoá.

### 5.4 Dùng trực tiếp với DataLoader

```python
from pathlib import Path
from torch.utils.data import DataLoader
from bevlane.dataset import BevLaneDataset

root = Path(r"D:\navsim_workspace\meteor_mini")
scenes = root.joinpath("scenes.txt").read_text().split()

dataset = BevLaneDataset(
    str(root),
    scenes,
    gt_key="gt",  # mặc định hiện trỏ đến map-derived target; gt_map cũng được
    with_agenttraj=True,
    with_ego=True,
    with_command=True,
    # Chỉ bật nếu converter đã chạy với flag tương ứng:
    with_depth=False,
    with_lidarbev=False,
)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
(imgs, K, T_cam_ego, gt, boxes, box_count,
 agent_traj, traj_valid, ego, command) = next(iter(loader))
```

### 5.5 Tạo trajectory vật thể từ NAVSIM tracking

`track_tokens` được nối trong từng `scene_token`, không nối qua biên scene.
Tâm box tương lai ở hệ ego tương lai được chuyển qua global rồi đưa về hệ ego
**của frame hiện tại**. Mỗi target có boxes `[64,6]`, trajectory offset
`[64,6,2]` tại `+0,5..+3,0 s` và validity `[64,6]`:

```powershell
py -3.12 -m bevlane.extract_navsim_agent_traj `
  --data-root "D:\navsim_workspace\dataset" `
  --out "D:\navsim_workspace\meteor_mini" `
  --split mini `
  --workers 4
```

Script ghi atomic, hỗ trợ resume, tạo `manifest.before_agent_traj.json` trước
khi thêm `agent_traj` vào manifest. Toàn bộ mini hiện đã materialize đủ 51.867
target.

### 5.6 Chia train/validation

Phải chia theo `<log-name>`/recording, không chia ngẫu nhiên theo frame. Các
frame liền nhau ở 2 Hz gần như trùng cảnh; chia theo frame gây leakage rất lớn.
`scenes.txt` chứa 64 recording để tạo `train_scenes.txt` và `val_scenes.txt`.
Sau đó dùng các option sẵn có của trainer:

```powershell
py -3.12 -m bevlane.train `
  --root "D:\navsim_workspace\meteor_mini" `
  --gt-key gt_map `
  --driving-command-source raw `
  --traj-w 0.3 `
  --train-list "D:\navsim_workspace\train_scenes.txt" `
  --val-scenes-file "D:\navsim_workspace\val_scenes.txt" `
  ...
```

`--traj-w 0.3` chỉ là điểm bắt đầu cho thí nghiệm; phải so A/B với `0.0` trước
khi chọn recipe. `--driving-command-source raw` dùng NAVSIM one-hot
`[left,straight,right,unknown]`, đổi sang METEOR `[straight,left,right]`; unknown
trở thành `[0,0,0]`. Mặc định `derived` được giữ để không phá dataset cũ.

Không bật loss cần target chưa tạo. `gt`/`gt_map` chỉ có năm class ở bảng trên;
depth và LiDAR BEV chỉ tồn tại nếu đã chạy các flag tương ứng. Một training
recipe hợp lệ phải ghi rõ head/loss nào có supervision và không tự diễn giải
class 4 như nhãn painted lane-line.

## 6. Shape và thứ tự output của `BevLaneDataset`

Base tuple luôn bắt đầu bằng:

| Vị trí | Tensor | Shape mỗi sample |
|---:|---|---:|
| 0 | `imgs` | `[8,3,432,768]` |
| 1 | `K` | `[8,3,3]` |
| 2 | `T_cam_ego` | `[8,4,4]` |
| 3 | `gt` | `[800,500]` |

Các flag nối tensor theo đúng thứ tự trong `bevlane/dataset.py`:

- `with_depth=True` → depth `[8,108,192]`;
- `with_boxdet=True` → boxes `[64,6]`, count scalar;
- `with_agenttraj=True` → boxes `[64,6]`, count, trajectory `[64,6,2]`,
  validity `[64,6]`;
- `with_ego=True` → ego vector `[17]`;
- `with_lidarbev=True` → `[4,400,250]`;
- `with_command=True` → command `[3]` và luôn được nối **cuối tuple** để không
  làm dịch index của các target cũ.

Trainer hiện tại unpack theo thứ tự này; không tự ý đổi thứ tự trong converter.

## 7. Giới hạn đã biết

1. Map target là hình học HD map, không phải cảm nhận của camera; class 4 và 6
   là proxy hình học. Cần so với ảnh/LiDAR hoặc nhãn quan sát độc lập nếu dùng
   để kết luận chất lượng lane perception.
2. Distortion coefficient được lưu vào manifest để truy vết nhưng pipeline hiện
   dùng pinhole `K`, giống adapter NAVSIM hiện tại; nó chưa remap fisheye.
3. Hai side camera NAVSIM đi vào hai slot narrow lịch sử của METEOR; đây là
   cross-rig zero-shot mapping, không phải calibration tương đương.
4. `gt_boxes` dùng trực tiếp trong local merged-LiDAR frame; static classes bị
   loại vì head METEOR chỉ có Vehicle/VRU.
5. `--with-depth` tạo sparse depth. Nó không thay thế dense depth pipeline dựa
   trên panoptic segmentation của dataset nội bộ.
6. Việc load batch thành công không đồng nghĩa model đã đạt metric tốt trên
   NAVSIM; cần checkpoint inference và evaluation riêng.

## 8. Nguồn contract

- NAVSIM v2.2 `navsim/common/dataclasses.py`: camera, LiDAR, ego, annotation và
  scene contract.
- NAVSIM v2.2 `navsim/common/dataloader.py`: cách pickle logs được chia thành
  scene và sensor path được resolve.
- METEOR `bevlane/dataset.py`: manifest, normalization và tuple output.
- METEOR `navsim_meteor/geometry.py`: camera order, resize/intrinsic và
  camera-to-LiDAR inversion dùng cho adapter zero-shot.
