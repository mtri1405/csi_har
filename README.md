# Nhận biết hiện diện và chuyển động bằng Wi-Fi CSI

Hệ thống gồm **1 AP phát UDP → 1 STA thu CSI → máy tính lưu CSV, huấn luyện Random Forest và dự đoán hai phase**. CSI phản ánh kênh truyền vô tuyến, không phải nội dung gói UDP.

## Luồng hoạt động

1. AP tạo mạng `CSI_SENSOR_AP`, kênh 6, HT20; gửi UDP 256 byte đến `192.168.4.2:3333` mỗi 40 ms (25 Hz danh định). Chỉnh `SEND_INTERVAL_MS` trong `csi_send/main/udp_sender.c` trước khi build/flash. Khoảng phát dùng lịch FreeRTOS cố định; tốc độ CSI thực tế vẫn cần đo.
2. STA dùng IP tĩnh `192.168.4.2`, tắt tiết kiệm điện, mở socket UDP và bật CSI. Callback lọc MAC AP, HT20 không STBC, CSI dài 256 byte; đưa vào queue rồi in CSV qua UART 2.000.000 baud. Không xử lý nặng trong Wi-Fi callback.
3. Máy tính lưu vào `tools/Data/<label>/`. Ba nhãn: `empty` (không người), `standing` (có người đứng yên), `walking` (có người đi lại).
4. Tách **nguyên recording** thành train/validation/test trước khi tạo cửa sổ. Baseline chỉ lấy từ các recording empty thuộc train.
5. Chuyển IQ thành biên độ, chọn 107 sóng mang (51 LLTF + 56 HT-LTF), nội suy theo timestamp, chuẩn hóa median riêng từng LTF, làm mượt Savitzky–Golay trong từng cửa sổ và lấy 13 đặc trưng.
6. Random Forest phase 1: `empty` / `present`. Chỉ khi có người mới chạy Random Forest phase 2: `standing` / `walking`. Cả hai model chỉ fit tập train; validation và test dùng đánh giá riêng.
7. Model đóng gói hai RF, baseline, thứ tự đặc trưng và cấu hình tiền xử lý. Realtime tự đọc cấu hình này, không cần nhập lại tần số hoặc layout.

Mặc định cửa sổ 2 giây, bước 0,5 giây, tần số 25 Hz. Timestamp phần cứng 32 bit được mở rộng khi tràn; reset hoặc đảo thứ tự trong recording sẽ bị từ chối. Realtime xóa cửa sổ khi reset, đổi kênh hoặc khoảng mất gói >0,25 giây. Không nội suy qua khoảng mất dữ liệu dài. Không tự giảm tần số khi tốc độ nguồn vượt 130% tần số cấu hình vì chưa có bộ lọc chống alias dùng chung cho offline/realtime.

## Cấu trúc

```text
Project/
├── csi_send/               # Firmware AP phát UDP
├── csi_recv/               # Firmware STA thu CSI
├── tools/
│   ├── run.py              # Điểm chạy chung cho web và dòng lệnh
│   ├── start_web.bat       # Mở web trên Windows
│   ├── requirements.txt
│   ├── csi/                # Mã Python đang dùng
│   │   ├── paths.py        # Đường dẫn dữ liệu và đầu ra dùng chung
│   │   ├── ports.py        # Liệt kê COM
│   │   ├── protocol.py     # Đọc frame serial, kiểm tra IQ và timestamp
│   │   ├── recording.py    # Ghi dữ liệu vào CSV
│   │   ├── signal.py       # LLTF/HT-LTF, baseline, đặc trưng, dataset
│   │   ├── training.py     # Random Forest hai phase và đánh giá
│   │   ├── realtime.py     # Dự đoán trực tiếp
│   │   ├── viewer.py       # Dữ liệu biểu đồ CSV/trực tiếp
│   │   ├── audit.py        # Kiểm tra chất lượng CSV
│   │   └── webapp.py       # API và quản lý tác vụ web
│   ├── web/                # HTML và biểu đồ JavaScript
│   ├── tests/              # Kiểm thử tự động
│   ├── Data/               # CSV theo nhãn empty/standing/walking
│   ├── reports/            # Dataset, model, metrics tự sinh
│   └── venv/               # Môi trường Python cục bộ
└── README.md
```

Các thư mục firmware giữ cấu trúc ESP-IDF. Dữ liệu và kết quả được giữ tại đường dẫn cũ; việc sắp xếp mã không yêu cầu huấn luyện lại model. Các thử nghiệm CNN/XGBoost, script cũ trong `remove` và báo cáo lỗi thời đã được loại khỏi luồng mã nguồn.

## Cài đặt và mở giao diện

Cần Python 3.10+ (đã kiểm thử với môi trường Python 3.12 hiện có). Từ thư mục gốc:

```powershell
py -3.12 -m venv tools/venv
tools/venv/Scripts/python.exe -m pip install -r tools/requirements.txt
tools/venv/Scripts/python.exe -X utf8 tools/run.py web
```

Hoặc chạy `tools/start_web.bat` nếu môi trường đã được cài. Mở http://127.0.0.1:8765.

Giao diện cho phép chọn COM, ghi dữ liệu, kiểm tra CSV, build dataset, huấn luyện, xem báo cáo validation/test và dự đoán trực tiếp. Chỉ chạy một tác vụ tại một thời điểm để tránh tranh chấp COM hoặc ghi đè đầu ra. Có nút dừng; log được giới hạn dung lượng. Khi không còn CSI hợp lệ, giao diện báo mất tín hiệu.

## Quan sát tín hiệu CSI trên web

Mở mục **Quan sát tín hiệu CSI**:

- **File CSV đã ghi:** chọn file trong `tools/Data`, bấm **Mở tín hiệu CSV**. Trang hiển thị biên độ của sóng mang được chọn, biên độ trung bình và bản đồ nhiệt 107 sóng mang. Có thể xem CSV ngay cả khi chưa huấn luyện model. Tối đa 1.000 mẫu trải đều toàn bộ phiên được vẽ; tổng mẫu, thời lượng, tần số và khoảng mất gói vẫn tính từ dữ liệu gốc. Việc giảm mẫu này chỉ phục vụ hiển thị, không sửa CSV hoặc dữ liệu train.
- **Trực tiếp từ STA:** chọn COM, bấm **Bắt đầu quan sát**, dùng **Dừng quan sát** để nhả cổng. Không cần model. Dừng ghi/dự đoán trước vì chỉ một tác vụ được dùng COM tại một thời điểm. Luồng xem trước giới hạn 10 mẫu/giây và giữ 200 mẫu gần nhất; tần số hiển thị ở chế độ này là tần số xem trước, không phải tốc độ thu thực tế của STA. Sau ba giây không có mẫu mới sẽ báo mất tín hiệu.
- Chọn **Biên độ gốc** hoặc **Chuẩn hóa median từng LTF**. Trục biên độ không phải dBm. Chỉ số 0–50 thuộc LLTF, 51–106 thuộc HT-LTF sau chọn sóng mang. Bản đồ nhiệt tự điều chỉnh thang màu theo đoạn đang xem; không so sánh màu giữa hai phiên như cùng một thang cố định.
- CSV sai layout, không có dữ liệu hợp lệ hoặc timestamp reset sẽ báo lỗi. Các khoảng mất tín hiệu lớn được ngắt nét trên biểu đồ. File lớn hơn 50 MB cần chia phiên để xem.

Sau khi cập nhật mã, khởi động lại `tools/run.py web` rồi tải lại trang để có API và giao diện mới.

## Thu dữ liệu và đánh giá đúng

- Flash cả hai board bằng ESP-IDF, kiểm tra target thực tế trước: `idf.py set-target esp32`, `idf.py build`, `idf.py -p COMx flash`. Chạy trong từng thư mục firmware từ ESP-IDF terminal. Nếu board là S3, chọn target đúng và kiểm chứng lại layout; không mặc định rằng dữ liệu ESP32 và S3 giống nhau.
- Kiểm tra cấu hình console thực tế trong sdkconfig: receiver phải xuất đúng 2.000.000 baud, CSI phải bật. sdkconfig.defaults không tự thay thế mọi giá trị của sdkconfig đã có.
- Ghi **ít nhất 3 recording độc lập cho mỗi nhãn**, tối thiểu 9 file, để có train/val/test đủ lớp. Nên thu nhiều phiên hơn, mỗi phiên 30–60 giây, thay đổi vị trí/người/thời điểm phù hợp mục tiêu. Không cắt một phiên thành nhiều file để giả lập các phiên độc lập.
- `standing` ở đây là người đứng yên; `walking` chỉ đại diện cho đi lại. Muốn bao quát mọi hành động cần thu thêm dữ liệu các hành động và thiết kế nhãn trước, không đổi tên dữ liệu falling thành walking.
- Web chia file theo thứ tự tên vào train/val/test luân phiên; đây là mức tối thiểu. Với dữ liệu nhiều ngày/người, hãy viết manifest thủ công để tách theo ngày/người/phòng, tránh các phiên gần nhau lọt sang các tập khác nhau.
- Build dataset từ nút trên web, sau đó Train model. Nếu dữ liệu thiếu lớp, trùng file/nội dung, sai timestamp hoặc sai layout, hệ thống báo lỗi để xử lý dữ liệu trước.
- Chọn COM dự đoán; đóng serial monitor/recorder đang giữ cổng đó. Lần dự đoán đầu tiên cần đủ một cửa sổ.

Chạy thủ công với manifest CSV gồm `session_id,label,split,rx_csv` (đường dẫn tương đối tính từ manifest):

```powershell
tools/venv/Scripts/python.exe -X utf8 tools/run.py prepare --manifest manifest.csv --out tools/reports/features --fs 25
tools/venv/Scripts/python.exe -X utf8 tools/run.py train --features-dir tools/reports/features --out tools/reports/two_phase_rf --trees 400
tools/venv/Scripts/python.exe -X utf8 tools/run.py predict --port COM9 --model tools/reports/two_phase_rf/model.joblib
```

`metrics.json` gồm phase 1, phase 2 trên dữ liệu có người thật, và toàn bộ chuỗi hai phase. Đánh giá phase 2 riêng không bao gồm lỗi bỏ sót người của phase 1; cần đọc cả end-to-end. Thứ tự confusion matrix: phase 1 `[0=empty,1=present]`, phase 2 `[1=standing,2=walking]`, end-to-end `[0,1,2]`. Xác suất RF chưa được hiệu chỉnh, không xem là độ tin cậy tuyệt đối.

## Kiểm chứng và giới hạn

Hiện có 9 file CSV thử nghiệm: 3 empty, 3 standing, 3 walking. Nhãn của các file ghi ngẫu nhiên chỉ phục vụ thử luồng phần mềm; không dùng độ chính xác trên các file đó để kết luận chất lượng nhận diện thực tế.

Kiểm thử bao gồm tạo dataset → fit hai RF → đánh giá → phát lại streaming, so sánh đặc trưng offline/realtime, tràn/reset timestamp, mất gói, trình xem CSI, API web và đường dẫn lệnh sau sắp xếp thư mục. Dữ liệu tổng hợp chỉ kiểm tra phần mềm.

Chưa build/flash firmware trong phiên kiểm tra vì môi trường hiện tại không có `idf.py`/`IDF_PATH`; chưa kiểm thử trên hai board hoặc xác nhận tần số RF thực tế. Lọc MAC/HT/độ dài không chứng minh từng mẫu CSI tương ứng duy nhất một datagram UDP: driver có thể phát CSI cho khung HT khác/retry từ cùng AP. Cần đo rate và kiểm tra môi trường mạng chuyên dụng trên thiết bị. Nhận biết người đứng yên phụ thuộc baseline, vị trí và môi trường; cần đánh giá ở các phiên độc lập trước khi kết luận đạt yêu cầu.

```powershell
tools/venv/Scripts/python.exe -X utf8 tools/run.py audit
tools/venv/Scripts/python.exe -m unittest discover -s tools/tests -p test_*.py
```

Tham khảo định dạng CSI, thứ tự IQ và hạn chế 4 byte đầu: [Espressif CSI documentation](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi-driver/wifi-vendor-features.html). LLTF đã loại carrier 0–1; layout HT-LTF 128 độc lập bị từ chối nếu cờ first_word_invalid ảnh hưởng carrier được chọn.

## Lệnh tiện ích

```powershell
tools/venv/Scripts/python.exe -X utf8 tools/run.py ports
tools/venv/Scripts/python.exe -X utf8 tools/run.py record --port COM9 --action empty --duration 30
tools/venv/Scripts/python.exe -X utf8 tools/run.py monitor --port COM9
tools/venv/Scripts/python.exe -X utf8 tools/run.py plot --csv tools/Data/empty/TEN_FILE.csv --out tools/reports/heatmap.png
tools/venv/Scripts/python.exe tools/run.py --help
```

Các lệnh dùng chung gói `tools/csi`; không chạy trực tiếp từng file bên trong gói. Dùng `<lệnh> --help` để xem tham số. Dữ liệu luôn lưu theo thư mục dự án dù khởi chạy từ thư mục khác. Sau khi sắp xếp mã, dừng web phiên cũ và mở lại bằng `tools/start_web.bat`.
