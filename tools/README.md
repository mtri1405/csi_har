# CSI Human Action Recognition

Dự án đọc tín hiệu Channel State Information (CSI) từ ESP32, xử lý dữ liệu CSI
và huấn luyện mô hình CNN để nhận diện hành động của con người.

## Yêu cầu môi trường

- Windows 10/11
- Python 3.10 trở lên
- ESP32 phát dữ liệu CSI qua cổng Serial

## Cài đặt môi trường ảo

Mở PowerShell tại thư mục dự án:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn việc chạy script kích hoạt môi trường, chạy lệnh sau một
lần ở phạm vi tài khoản hiện tại rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Khi hoàn tất, có thể thoát môi trường ảo bằng:

```powershell
deactivate
```

## Quy trình thực hiện

### Cấu hình CSI cố định

Workflow nhận 64 subcarrier HT20, loại bỏ 17 vị trí cố định và dùng PCA còn
10 thành phần cho model:

| Thành phần | Giá trị bắt buộc |
| --- | --- |
| CSI complex sample mỗi frame | `64` |
| Giá trị I/Q trong CSV `data` | `128` (`I0,Q0,...,I63,Q63`) |
| Biên độ sau parse | `(frames, 64)` |
| Sau mask subcarrier | `(frames, 47)` |
| Cửa sổ model | `(60 frames, 10 PCA components)` |
| Realtime checkpoint indices | 47 chỉ số được giữ lại |

Không đổi riêng một bước sang layout khác. Nếu thay đổi mask,
`PCA_COMPONENTS` hoặc `TARGET_SUBCARRIERS` trong `csi_preprocessing.py`, phải
tạo lại dataset và train lại checkpoint trước khi chạy realtime. Firmware dùng
`CSI_SUBCARRIERS` và `CSI_IQ_BYTES` trong
`shared/csi_proto/include/csi_proto.h` làm nguồn sự thật.

## Chạy giao diện web

Giao diện web không thể tự kích hoạt `venv` hoặc chạy tiến trình Python trực
tiếp vì giới hạn bảo mật của trình duyệt. Dự án cung cấp local backend để làm
việc đó bằng Python trong `venv`:

```powershell
.\start_csi_web.bat
```

Hoặc chạy thủ công:

```powershell
.\venv\Scripts\Activate.ps1
python csi_web_server.py
```

Sau đó mở `http://127.0.0.1:8765/`. Giao diện cho phép:

- Chọn cổng Serial và baudrate.
- Chọn nhãn hành động, thời gian ghi và thời gian đếm ngược.
- Phát âm thanh khi đếm ngược, bắt đầu và kết thúc ghi.
- Chạy tạo dataset, train CNN và realtime prediction.
- Xem CSV hoặc CSI live trên cùng trang, chọn subcarrier 0-63 và tinh chỉnh
  chế độ/tham số bộ lọc mà không sửa file dữ liệu.
- Theo dõi log stdout/stderr của tiến trình Python và dừng realtime prediction.

Nếu không muốn dùng web, có thể chạy trực tiếp từng script bằng Python trong
venv như hướng dẫn bên dưới.

### 1. Thu thập dữ liệu CSI

Có thể chạy với tham số từ terminal, không cần chỉnh source:

```powershell
python record_to_csv.py --port COM8 --baudrate 2000000 --action standing --duration 30
```

Các tham số đều có giá trị mặc định, vì vậy `python record_to_csv.py` vẫn hoạt
động như trước.

Dữ liệu được lưu tại `Data/<ACTION_LABEL>/`. Mỗi file CSV bắt đầu bằng header:

```text
type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data
```

Các dòng sau là dữ liệu CSI nhận được từ ESP32.

### 2. Tạo dataset cho CNN

Đặt dữ liệu theo các thư mục nhãn mà `dataset_pipeline.py` khai báo
(`empty`, `walking`, `standing`), sau đó chạy:

```powershell
python dataset_pipeline.py
```

Lệnh này tạo file `csi_dataset.npz`.

Trong bước này, pipeline loại bỏ mask đã cấu hình, lọc theo thời gian, fit PCA
10 thành phần chỉ trên các window thuộc training set, sau đó dùng cùng PCA
transform cho mọi window và chuẩn hóa từng cửa sổ. Chi tiết được mô tả trong
[SUBCARRIER_SELECTION.md](./SUBCARRIER_SELECTION.md).

Mỗi file được xử lý độc lập và được chia thành các window liên tiếp dài
`60 frame`, không chồng lấn (`stride = 60`). Sau đó các file của từng label
được xáo trộn bằng seed cố định và chia đều theo file với tỷ lệ gần
`60% train`, `20% validation`, `20% test`. Tất cả window thuộc cùng một file
luôn nằm trong cùng một split; không có file nào bị chia giữa train,
validation và test. Các index split được lưu trong `csi_dataset.npz`, nên bước
train sử dụng chính xác cùng phép chia đã dùng khi fit PCA.

### 3. Huấn luyện mô hình

```powershell
python train.py
```

Mô hình được huấn luyện bằng Focal Loss (`gamma = 2.0`) thay cho
`CrossEntropyLoss`. Trọng số lớp được dùng làm `alpha`, giúp các mẫu khó phân
loại có ảnh hưởng lớn hơn trong quá trình cập nhật, đặc biệt khi Standing và
Empty có đặc trưng gần nhau. Checkpoint lưu lại cấu hình loss đã dùng.

Mô hình tốt nhất được lưu thành `csi_cnn_model.pth`.

Checkpoint chứa cả trọng số và metadata preprocessing (mask subcarrier, kích
thước cửa sổ, tần số mẫu và mapping nhãn). Nếu đã cập nhật code, cần chạy lại
`dataset_pipeline.py` trước `train.py`; predictor sẽ từ chối checkpoint cũ
không có metadata để tránh suy luận sai cấu hình.

### 4. Nhận diện theo thời gian thực

Đảm bảo ESP32 đang kết nối đúng cổng Serial và cập nhật `PORT` trong
`realtime_predict.py`, sau đó chạy:

```powershell
python realtime_predict.py --port COM8 --baudrate 2000000
```

Realtime kiểm tra tốc độ frame, làm mượt xác suất của 5 cửa sổ gần nhất và
hiển thị `Chưa xác định` khi độ tin cậy đã làm mượt dưới 60%. Chi tiết các
thay đổi xem tại
[REALTIME_PREDICTION_CHANGES.md](./REALTIME_PREDICTION_CHANGES.md).

## Công cụ phân tích CSI

`csi_data_read_parse.py` và `visualize_csi.py` là các công cụ trực quan
hóa/phân tích CSI có sẵn trong repo. Cả hai đều kiểm tra layout 64 subcarrier.
Có thể chạy bằng:

```powershell
python csi_data_read_parse.py
```

Để visualize CSI trực tiếp từ serial, chạy công cụ sau từ thư mục `tools`:

```powershell
python visualize_csi.py --port COM8 --baudrate 2000000
```

Cửa sổ realtime hiển thị dữ liệu raw, heatmap và dữ liệu sau bộ lọc. Có thể đổi
cửa sổ quan sát `1s`, `5s`, `15s`, thay đổi cổng/baudrate và chọn `Không lọc`, `Hampel`,
`Butterworth`, `Hampel + Butterworth` hoặc `Full pipeline` ngay trong cửa sổ.
Có thể xem tối đa ba CSV cùng lúc bằng cách truyền nhiều đường dẫn:

```powershell
python visualize_csi.py --csv Data\empty\a.csv Data\walking\b.csv Data\standing\c.csv
```

Mỗi file có một hàng riêng. Biểu đồ raw hiển thị đồng thời toàn bộ 64
subcarrier, chia thành 8 nhóm bằng nhau (8 subcarrier/nhóm), mỗi nhóm có một
màu riêng. Mỗi plot có nút chọn CSV riêng. Chế độ này có bộ chọn thời gian
độc lập cho raw và dữ liệu đã PCA, cùng danh sách filter `Không lọc`, `Hampel`,
`Butterworth`, `Hampel + Butterworth` và `Full pipeline`. Có thể chọn `Tất cả`
hoặc từng nhóm subcarrier bằng bộ chọn nhóm; dùng nút `Lên`, `Xuống` hoặc
cuộn chuột trên vùng plot để chuyển giữa các lựa chọn file.
Vẫn có thể xem CSV đã ghi trước đó bằng:

```powershell
python visualize_csi.py --data-dir Data
```

Sau khi thay đổi các thông số lọc, cần tạo lại dataset và huấn luyện lại model:

```powershell
python dataset_pipeline.py
python train.py
```

## Cấu trúc thư mục chính

```text
record_to_csv.py       # Thu CSI và ghi CSV
dataset_pipeline.py    # Parse, lọc và tạo cửa sổ dataset
train.py               # Huấn luyện CNN
realtime_predict.py    # Nhận diện CSI theo thời gian thực
csi_visualization.py   # Parse/lọc dùng chung cho web và visualize_csi.py
csi_data_read_parse.py # Phân tích và trực quan hóa CSI legacy
csi_preprocessing.py   # Mask subcarrier, lọc, PCA và chuẩn hóa
csi_web.html           # Giao diện web tiếng Anh
csi_web_server.py      # Backend local chạy pipeline bằng venv
start_csi_web.bat      # Khởi động backend và mở trình duyệt
requirements.txt       # Dependency của dự án
```
Xem dữ liệu
python visualize_csi.py --port COM8 --baudrate 2000000 --model csi_cnn_model.pth

python visualize_csi.py --port COM8 --baudrate 2000000