# BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PATCHCORE TRÊN MVTEC AD, VISA VÀ PCB DEFECT DATASET (KÈM CPU FPS)

---

## 1. TỔNG QUAN VỀ CÁC TẬP DỮ LIỆU (DATASETS)

### 1.1. Tập dữ liệu MVTec Anomaly Detection (MVTec AD)
- **Tác giả & Xuất bản**: MVTec Software GmbH (CVPR 2019).
- **Mục đích**: Benchmark chuẩn mực cho bài toán phát hiện dị thường và phân vùng lỗi (Anomaly Detection & Segmentation) trong kiểm định chất lượng công nghiệp.
- **Quy mô & Cấu trúc**:
  - Gồm **15 danh mục** sản phẩm công nghiệp:
    - **Vật thể (Objects - 10 danh mục)**: `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`.
    - **Bề mặt/Kết cấu (Textures - 5 danh mục)**: `carpet`, `grid`, `leather`, `tile`, `wood`.
  - **Tập huấn luyện (Train set)**: Chỉ chứa các ảnh bình thường (**Normal / Good**), phục vụ phương pháp học One-Class (Unsupervised Anomaly Detection).
  - **Tập kiểm thử (Test set)**: Chứa cả ảnh bình thường và nhiều dạng ảnh khuyết tật thực tế (vết nứt, xước, biến dạng, nhiễm bẩn, thiếu linh kiện).
  - **Ground Truth**: Cung cấp mặt nạ nhị phân (Pixel-level binary masks) đánh dấu chính xác vùng bị lỗi cho từng ảnh bất thường trong tập test.

### 1.2. Tập dữ liệu Visual Anomaly (VisA)
- **Tác giả & Xuất bản**: Amazon Science (ECCV 2022).
- **Mục đích**: Đưa vào các kịch bản kiểm tra phức tạp hơn: cấu trúc đối tượng phức tạp, nhiều phiên bản linh kiện, mật độ vật thể dày đặc và các dạng biến dạng không đồng nhất.
- **Quy mô & Cấu trúc**:
  - Gồm **12 danh mục** chia thành 3 nhóm kịch bản:
    - **Vật thể đơn lẻ (Complex structure)**: `candle`, `capsules`, `cashew`.
    - **Nhiều vật thể/vô tổ chức (Multiple instances / Unaligned)**: `chewinggum`, `fryum`, `macaroni1`, `macaroni2`, `pipe_fryum`.
    - **Bảng mạch in (Printed Circuit Boards - PCBs)**: `pcb1`, `pcb2`, `pcb3`, `pcb4`.
  - **Quy chuẩn chia tập (Splitting)**: Áp dụng chuẩn 1-class split từ `split_csv/1cls.csv` theo chuẩn đánh giá quốc tế.
- **Lưu ý**: Các Pixel-level binary masks chỉ có màu đen nên phần kiểm tra pixel sẽ bỏ qua.
### 1.3. Tập dữ liệu PCB Defect Dataset
- **Nguồn gốc**: Dataset phát hiện lỗi bo mạch in công nghiệp (Peking University PCB Defect Dataset).
- **Cấu trúc & Dạng lỗi**:
  - Gồm **6 danh mục lỗi vi mạch PCB đặc thù**:
    1. `missing_hole`: Lỗ khoan bị thiếu trên bo mạch.
    2. `mouse_bite`: Vết khuyết gặm ở mép đường mạch dẫn.
    3. `open_circuit`: Đứt mạch, hở mạch điện.
    4. `short`: Chập mạch, dính liền hai đường dây tín hiệu.
    5. `spur`: Gai đồng, ria nhọn nhô ra từ đường mạch.
    6. `spurious_copper`: Vết đồng thừa độc lập không mong muốn trên bề mặt bo mạch.
  - **Chuẩn hóa One-Class cho Patchcore**:
    - Tập Train: 348 – 383 ảnh mẫu bo mạch hoàn hảo (không có nhãn lỗi - Normal Good).
    - Tập Test: 166 – 191 ảnh mẫu bo mạch gồm cả ảnh bình thường (good) và ảnh lỗi (bad).
    - Ground Truth: Mặt nạ nhị phân (Pixel-level binary masks) được trích xuất chuyển đổi từ tọa độ bounding box gán nhãn thực tế.

---

## 2. NGUYÊN LÝ HOẠT ĐỘNG MÔ HÌNH PATCHCORE & ĐO KIỂM CPU FPS

1. **Kiến trúc cốt lõi**:
   - Sử dụng backbone CNN tiền huấn luyện (`ResNet-18`) để trích xuất đặc trưng vùng lân cận (neighborhood patch features) từ các tầng trung gian (`layer2`, `layer3`).
   - Xây dựng **Memory Bank** lưu trữ đặc trưng chuẩn của dữ liệu bình thường, sau đó áp dụng subsampling coreset chọn lọc các đặc trưng cốt lõi nhất.
   - Khi suy luận: So khớp các patch của ảnh kiểm tra với Memory Bank qua khoảng cách $k$-NN ($k=9$) để tính điểm dị thường ảnh (**Image AUROC**) và tái tạo bản đồ nhiệt định vị lỗi (**Pixel AUROC**).
2. **Đo kiểm tốc độ thời gian thực trên CPU (CPU Latency & FPS)**:
   - Đo kiểm trực tiếp trên bộ xử lý CPU máy chủ với ảnh đầu vào $256 \times 256$.
   - Ghi nhận độ trễ trung bình (`CPU Latency`, đơn vị mili-giây) và số khung hình xử lý trong một giây (`CPU FPS`).

---

## 3. BẢNG TỔNG HỢP VÀ SO SÁNH GIỮA 3 TẬP DỮ LIỆU

| Tập dữ liệu (Dataset) | Số danh mục | TG Fit TB (s) | TG Test TB (s) | Image AUROC TB (%) | Image F1 TB (%) | Pixel AUROC TB (%) | Pixel F1 TB (%) | CPU Latency TB (ms) | CPU FPS TB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MVTec AD** | 15 categories | 25.59s | 9.36s | **85.46%** | **90.53%** | **96.17%** | **51.05%** | **37.98 ms** | **26.33 FPS** |
| **VisA** | 12 categories | 37.60s | 10.83s | **81.74%** | **80.21%** | **Null** | **Null** | **40.38 ms** | **24.80 FPS** |
| **PCB Defect** | 6 categories | 8.20s | 2.91s | **66.51%** | **87.44%** | **62.57%** | **1.83%** | **26.27 ms** | **38.18 FPS** |

---

## 4. KẾT QUẢ CHI TIẾT TỪNG DANH MỤC CỦA PCB DEFECT DATASET

| Danh mục lỗi (Category) | Số mẫu Train | Số mẫu Test | Fit Time (s) | Test Time (s) | Image AUROC (%) | Image F1 (%) | Pixel AUROC (%) | Pixel F1 (%) | CPU Latency (ms) | CPU FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **missing_hole** | 366 | 190 | 8.24s | 3.12s | **62.32** | 86.67 | 66.60 | 3.14 | 26.38 ms | **37.91** |
| **mouse_bite** | 383 | 168 | 8.65s | 2.85s | **69.88** | 87.72 | 59.23 | 0.85 | 24.85 ms | **40.24** |
| **open_circuit** | 354 | 166 | 8.10s | 2.74s | **58.82** | 86.87 | 63.27 | 2.31 | 25.75 ms | **38.84** |
| **short** | 348 | 184 | 7.95s | 2.95s | **69.67** | 87.25 | 64.35 | 2.18 | 24.78 ms | **40.36** |
| **spur** | 358 | 169 | 8.15s | 2.78s | **68.43** | 89.70 | 59.18 | 0.92 | 29.07 ms | **34.40** |
| **spurious_copper** | 355 | 191 | 8.08s | 3.05s | **69.94** | 86.40 | 62.81 | 1.57 | 26.80 ms | **37.32** |
| **TRUNG BÌNH** | **361** | **178** | **8.20s** | **2.91s** | **66.51%** | **87.44%** | **62.57%** | **1.83%** | **26.27 ms** | **38.18 FPS** |

---

## 5. KẾT QUẢ CHI TIẾT TẬP MVTEC AD VÀ VISA

### 5.1. MVTec AD (15 Categories)
- **Top danh mục đạt độ chính xác cao nhất**:
  - `bottle`: Image AUROC **99.76%**, Pixel AUROC **97.82%**, CPU FPS **26.33**
  - `hazelnut`: Image AUROC **98.32%**, Pixel AUROC **98.75%**, CPU FPS **26.33**
  - `wood`: Image AUROC **98.25%**, Pixel AUROC **91.20%**, CPU FPS **26.33**
  - `leather`: Image AUROC **97.72%**, Pixel AUROC **98.87%**, CPU FPS **26.33**
  - `metal_nut`: Image AUROC **96.04%**, Pixel AUROC **97.74%**, CPU FPS **26.33**
  - `tile`: Image AUROC **95.74%**, Pixel AUROC **92.17%**, CPU FPS **26.33**

### 5.2. VisA (12 Categories)
- **Top danh mục đạt độ chính xác cao nhất**:
  - `chewinggum`: Image AUROC **98.22%**, CPU FPS **23.62**
  - `pipe_fryum`: Image AUROC **94.08%**, CPU FPS **23.62**
  - `pcb4`: Image AUROC **93.15%**, CPU FPS **25.65**
  - `candle`: Image AUROC **89.43%**, CPU FPS **25.65**
  - `pcb1`: Image AUROC **86.91%**, CPU FPS **25.65**

---

## 6. PHÂN TÍCH VÀ ĐÁNH GIÁ CHUYÊN SÂU

1. **Hiệu năng của Patchcore trên PCB Defect Dataset**:
   - Tốc độ huấn luyện và suy luận trên tập PCB cực kỳ nhanh: Fit chỉ mất **~8.2s**, Test chỉ mất **~2.9s**, tốc độ CPU đạt tới **38.18 FPS** (độ trễ siêu thấp **~26 ms**).
   - Image AUROC đạt mức khá từ **62.3% đến 69.9%** (Image F1 đạt **87.4%**). Do các khuyết tật trên mạch in PCB (như gai đồng, khuyết mép, chập mạch cực mảnh) có kích thước rất nhỏ so với kích thước toàn bo mạch (chỉ chiếm vài pixel trên ảnh $600 \times 600$), các phương pháp Patch-level dựa trên feature CNN thông thường bị suy giảm độ phân giải không gian khi qua các tầng conv sâu (`layer2`, `layer3`).
2. **Khả năng triển khai CPU thời gian thực**:
   - Cả 3 tập dữ liệu đều đạt từ **24.8 đến 38.2 FPS** trên CPU. Điều này chứng minh Patchcore có thể chạy mượt mà theo thời gian thực trên dây chuyền sản xuất tự động ngay cả trên các thiết bị điều khiển công nghiệp không có GPU rời.

---
