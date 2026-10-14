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

Trong bước này, pipeline loại bỏ các subcarrier biên, subcarrier DC ở giữa và
các cột có giá trị 0 trong toàn bộ file. Chi tiết lựa chọn được mô tả trong
[SUBCARRIER_SELECTION.md](./SUBCARRIER_SELECTION.md).

### 3. Huấn luyện mô hình

```powershell
python train.py
```

Mô hình tốt nhất được lưu thành `csi_cnn_model.pth`.

### 4. Nhận diện theo thời gian thực

Đảm bảo ESP32 đang kết nối đúng cổng Serial và cập nhật `PORT` trong
`realtime_predict.py`, sau đó chạy:

```powershell
python realtime_predict.py --port COM8 --baudrate 2000000
```

## Công cụ phân tích CSI

`csi_data_read_parse.py` là công cụ trực quan hóa/phân tích CSI có sẵn trong
repo. Có thể chạy bằng:

```powershell
python csi_data_read_parse.py
```

## Cấu trúc thư mục chính

```text
record_to_csv.py       # Thu CSI và ghi CSV
dataset_pipeline.py    # Parse, lọc và tạo cửa sổ dataset
train.py               # Huấn luyện CNN
realtime_predict.py    # Nhận diện CSI theo thời gian thực
csi_data_read_parse.py # Phân tích và trực quan hóa CSI
csi_preprocessing.py   # Mask subcarrier biên, DC và cột zero
csi_web.html           # Giao diện web tiếng Anh
csi_web_server.py      # Backend local chạy pipeline bằng venv
start_csi_web.bat      # Khởi động backend và mở trình duyệt
requirements.txt       # Dependency của dự án
```