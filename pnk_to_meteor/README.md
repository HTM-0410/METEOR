# PNK raw → METEOR Layer 2, 9 head

Thư mục này chứa mã tự chạy, không cần import từ checkout METEOR gốc. Đầu ra là sample Layer 2 cho 9 head đang dùng: 1, 2, 3, 5, 7, 8, 9, 10, 12. Head 4 (unknown object), 6 (2D detection), 11 (traffic light) đã loại khỏi phạm vi theo quyết định hiện tại. Đây là bộ **candidate/pseudo GT**; PASS kiểm tra schema và DataLoader không tương đương nghiệm thu độ chính xác nhãn.

## Yêu cầu dữ liệu và máy

- Python 3.12, đủ dung lượng cho raw và nhiều profile trung gian. Nên dùng SSD, GPU NVIDIA cho hai teacher panoptic; CPU có thể chạy nhưng rất chậm. Bản đã chạy ở đây dùng torch `2.12.1+cu130`, torchvision `0.27.1+cu130`, transformers `5.7.0`. Những phiên bản khác chưa được xác nhận với gói này.
- Đặt raw và metadata cùng dưới `source_root`:

```text
source_root/
  PNKData/                                      # ảnh JPEG, LiDAR LAZ, NAV...
  PNKData_meta/
    label/2026_08_14_vf6_01_02/
    file_csv/2026_08_14_vf6_01_02.csv
    Calib_2/VF6_01_Intrinsics.json
    Calib_2/VF6_01_Extrinsics_By_Dates.json
```

Tên metadata trên đang cố định theo bộ PNK được cung cấp. Máy khác phải có đúng snapshot/schema này, hoặc sửa `src/scripts/prepare_pnk_clean.py` và kiểm tra lại mapping. Ảnh, LiDAR và metadata phải giữ nguyên qua lần chạy; output ghi đường dẫn tuyệt đối tới raw và profile trước đó. Vì vậy **không di chuyển/xóa raw hoặc các profile trung gian sau khi tạo output**. Chạy lại pipeline nếu đổi ổ đĩa/mount.

## Cài đặt (PowerShell)

```powershell
cd <duong_dan>/pnk_to_meteor
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Cài torch + torchvision bằng lệnh do bộ chọn chính thức PyTorch cung cấp cho GPU/CPU máy này:
# https://docs.pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -c "import torch, torchvision, transformers, cv2, laspy, lazrs; print(torch.__version__, torch.cuda.is_available())"
Copy-Item config.example.json config.json
# Sửa source_root và work_root trong config.json; work_root phải trống và khác source_root.
python run.py --config config.json --dry-run
python run.py --config config.json --check
python run.py --config config.json
```

Linux/macOS: tạo venv bằng `python3.12 -m venv .venv`, kích hoạt `source .venv/bin/activate`, chạy các lệnh `python` tương tự. Đây là hướng dẫn chuyển máy; lần chạy đầy đủ trên OS khác Windows chưa được kiểm chứng. Cần Internet lần đầu để tải hai checkpoint Hugging Face đã ghim revision: Cityscapes Mask2Former `a87607429f7474fd1e2d1d55d6a4ce18a893a526` và Vistas MaskFormer `ae4b8c2590c0a090fc32d5c217d78738a2dd4b19`; các lần sau dùng cache của Hugging Face.

`panoptic_batch_size=8` đã được thử trên máy hiện tại; giảm xuống 2 hoặc 4 nếu thiếu VRAM. `target_workers` điều khiển các bước CPU. Pipeline luôn chạy toàn bộ dataset, không có cờ sample/max-frames vì các bước trajectory và temporal fusion cần tập liên tục.

## Flow xử lý chuẩn: raw PNK → sample Layer 2

```text
Ảnh 8 camera + LiDAR LAZ + NAV/IMU/xe + 3D box + calibration
  → kiểm tra và đồng bộ thời gian
  → chuẩn hóa hệ tọa độ, hiệu chỉnh camera, tạo dữ liệu LiDAR
  → suy luận nhãn ảnh Layer 1
  → ghép sample nền Layer 2
  → tạo nhãn 9 head theo quan hệ phụ thuộc
  → gắn mask, confidence, provenance
  → kiểm tra hình học/contract/DataLoader → sample đầu ra
```

### 1. Tiếp nhận và đồng bộ raw

Đọc danh sách scene, ảnh JPEG, LiDAR LAZ, NAV/IMU/tín hiệu xe, nhãn 3D và intrinsics/extrinsics. Chọn keyframe theo danh sách PNK, ghép cảm biến theo timestamp trong ngưỡng cho phép, kiểm tra file/calibration/shape và loại bản ghi không đạt. Chuẩn hóa geometry nhãn 3D, giữ track ID, class và provenance nguồn; raw chỉ được đọc. Ngữ nghĩa track ID và LiDAR deskew chưa được xác nhận độc lập.

### 2. Chuẩn hóa hình học và đầu vào cảm biến

Ánh xạ 8 camera PNK vào thứ tự camera METEOR; thống nhất phép biến đổi camera ↔ ego và LiDAR ↔ ego. Dùng calibration để khử distortion, resize ảnh về 432×768 và tính lại ma trận K đúng với ảnh mới. Từ LiDAR tạo đặc trưng BEV làm **input**; nó chưa phải nhãn occupancy. Với box 3D, kiểm tra geometry và vùng BEV, giữ cả box ít điểm LiDAR nếu còn hợp lệ và ghi confidence theo số điểm hỗ trợ. Kiểm tra timestamp, phép biến đổi camera vòng đi/về trên điểm kiểm thử và khả năng đọc sensor trước khi tạo nhãn tiếp theo.

### 3. Tạo nhãn ảnh Layer 1

Chạy Cityscapes Mask2Former trên ảnh đã hiệu chỉnh để tạo panoptic/semantic candidate, rồi ánh xạ sang slot class 2D của METEOR. Class không được teacher hỗ trợ phải để `ignore`; không suy đoán thành background. Đây là teacher bên ngoài, **không phải checkpoint panoptic CoMET gốc**. Front camera còn được chạy qua Vistas MaskFormer khi tạo ứng viên crosswalk, marking và parking ở bước BEV lane.

### 4. Ghép sample nền Layer 2

Tạo một manifest cho mỗi frame: ảnh 8 camera, K, T_cam_ego, LiDAR BEV, NAV/ego state, 3D box, nhãn ảnh và đường dẫn tới các target. Mọi target dùng cùng keyframe và hệ tọa độ ego. Tại bước này chỉ ghép dữ liệu đã đồng bộ; nhãn cho từng head được hoàn thiện theo các phụ thuộc dưới đây.

### 5. Tạo target cho 9 head

| Head | Cách tạo target | Validity/confidence được lưu |
|---|---|---|
| **3 — 3D detection** | Box nguồn hợp lệ trong vùng BEV, giữ track/class và geometry. | Tier theo số điểm LiDAR; box ít điểm không tự động bị xóa. |
| **5 — 2D semantic** | Ánh xạ panoptic/semantic Layer 1 sang class METEOR. | Pixel thuộc class chưa có teacher → `ignore`. |
| **2 — Metric depth** | Chiếu LiDAR lên 8 ảnh đã hiệu chỉnh; lấp lỗ nhỏ trên bề mặt tĩnh cùng class. | Tách pixel đo trực tiếp, pixel nội suy và pixel không hợp lệ. |
| **1 — BEV lane** | Hợp nhất road/sidewalk từ quan sát hiện tại với LiDAR, depth, semantic front camera, LiDAR intensity và đồng thuận giữa frame để tạo road, sidewalk, crosswalk, lane line, stop line, road edge, marking, parking. | Vùng không đủ chứng cứ → `255 ignore`; lưu support votes và confidence. |
| **7 — E2E driving** | Từ NAV/tín hiệu xe tạo speed, acceleration, steering, brake và 6 ego waypoint tương lai; suy route command từ quỹ đạo đã đi. | Validity theo frame; command không suy được → `unknown`. Command này là pseudo intent, không phải lệnh route planner. |
| **10 — Agent forecasting** | Ghép box theo track ID và thời gian, bù chuyển động ego, tạo vị trí tương lai 0,5–3 s. | Mask riêng cho từng agent/horizon; loại gap, class switch, timestamp lỗi và cuối scene. |
| **9 — Occupancy flow** | Tính dịch chuyển/vận tốc từ agent trajectory hợp lệ. | Chỉ supervise cell có motion target đáng tin. |
| **8 — 3D occupancy** | Voxel hóa LiDAR hiện tại, rồi chỉ điền voxel unknown tĩnh khi các lượt quét lân cận đồng thuận; bảo vệ vật thể động và voxel hiện tại. | `unknown` giữ riêng, không biến thành free; confidence phân biệt voxel hiện tại với voxel điền theo thời gian. |
| **12 — Area risk** | Tính lại risk **sau khi** occupancy cuối và agent future đã có. | Là target dẫn xuất; kế thừa giới hạn của Head 8 và 10. |

Các target không phải cảm biến đo trực tiếp được ghi rõ provenance. Road/sidewalk, depth và occupancy chỉ được mở rộng vào vùng thiếu chứng cứ khi quy tắc đồng thuận đạt ngưỡng; dữ liệu đo được của frame hiện tại được giữ nguyên. Khi huấn luyện phải dùng đúng mask và trọng số confidence; loader hiện chưa tự áp trọng số confidence của occupancy.

### 6. Kiểm tra và xuất sample

Kiểm tra số scene/frame, liên kết file, camera calibration, shape/dtype, class ID, box 3D, mask/confidence, tính bất biến của đo đạc gốc và khả năng đọc qua `BevLaneDataset`. Sample cuối chứa 15 tensor (16 khi bật depth confidence) cho 9 head. Head 4/6/11 không được đưa vào contract. Sau kiểm tra kỹ thuật vẫn cần QA hình ảnh theo scene/class và A/B training trước khi gọi bộ nhãn là train-ready.

`run.py` hiện triển khai flow này bằng 26 stage nội bộ và ghi `work_root/pnk_pipeline_state.json` sau mỗi stage thành công. Tên thư mục/schema có hậu tố phiên bản là chi tiết lưu trữ để tương thích với dữ liệu đã tạo; **chúng không phải các bước của flow chuẩn**. Xem câu lệnh thực thi bằng `python run.py --config config.json --dry-run`. Khi chạy lại cùng config, stage hoàn tất được bỏ qua. Nếu dừng giữa stage, kiểm tra output dở dang hoặc chọn `work_root` mới; không sửa checkpoint để bỏ qua xác minh.

## Đầu ra và kiểm tra

Thư mục cuối hiện do launcher xuất ra là `work_root/PNKData_layer2_9head_v7/`; các thư mục còn lại dưới `work_root` là dữ liệu trung gian phục vụ resume và đường dẫn tham chiếu. Trong output cuối, `dataset.json` mô tả contract, `task_availability.json` mô tả mức sẵn sàng từng head, `nine_head_verification.json` ghi kết quả verifier, và mỗi `<scene>/manifest.json` liên kết target của các frame. Dùng `--dry-run` để xem đầy đủ đường dẫn output theo config của máy.

```powershell
python run.py --verify-existing "D:/PNK_meteor_build/PNKData_layer2_9head_v7"
python src/scripts/pnk_9head_viewer.py --root "D:/PNK_meteor_build/PNKData_layer2_9head_v7" --cache "D:/PNK_meteor_build/viewer_cache" --port 8765
```

Verifier tạo/cập nhật `nine_head_verification.json`. Viewer tại `http://127.0.0.1:8765/`. Đối với METEOR training, dùng `PNKData_layer2_9head_v7` làm dataset root với `bevlane.dataset.BevLaneDataset` của checkout tương thích. Bản dataset loader nằm trong `src/bevlane/dataset.py`; copy/sync riêng vào checkout training nếu checkout đó chưa hỗ trợ `with_depth_confidence`, route command và profile v7. Không copy đè mù vào checkout có thay đổi khác. Contract sample đã xác minh gồm 15 tensor (hoặc 16 khi bật depth confidence): 8 ảnh, K, T, gt_map, depth, 2D segmentation, agent boxes/count/traj/valid, ego, occupancy, risk, lidar BEV, driving command.

## Giới hạn cần biết

- LiDAR deskew được **giả định đã có** theo chỉ đạo dự án; không có chứng cứ exporter xác nhận. GNSS lever arm, track-ID schema và camera slot mapping cần xác nhận từ nguồn.
- Pseudo-label lane class 3–8, depth nội suy, route command, multi-sweep occupancy và các head dẫn xuất cần QA thủ công phân tầng trước huấn luyện/đánh giá model. `unknown` route command phải dùng mask. 2D class mapping hiện thiếu class nên giữ ignore, không tự suy ra background.
- Source scripts có đường dẫn mặc định Windows từ môi trường cũ; launcher luôn truyền đường dẫn explicit. Không gọi từng module mà không truyền đầy đủ tham số.
- Hướng dẫn này đóng gói pipeline tiền xử lý và sample loader; không đóng gói checkpoint CoMET teacher hoặc mã huấn luyện đầy đủ của METEOR.
- Bộ code đóng gói đã vượt qua compile, import độc lập, kiểm tra raw/dependency và verifier trên profile v7 hiện có. Chưa chạy lại trọn 26 stage từ raw vào `work_root` mới; đó là kiểm thử tích hợp còn lại khi chuyển máy.

Báo cáo số liệu của snapshot hiện tại được giữ trong `STATUS.md` ở bản làm việc nội bộ; gói mã công khai chỉ chứa flow, hợp đồng dữ liệu và hướng dẫn chạy.
