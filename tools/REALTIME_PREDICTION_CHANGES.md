# Các thay đổi cơ chế nhận diện CSI realtime

## Mục tiêu

Các thay đổi này nhằm giảm sai lệch giữa dữ liệu lúc huấn luyện và dữ liệu
nhận trực tiếp từ ESP32, đồng thời làm cho kết quả hiển thị ổn định hơn. File
`realtime_predict.py` đang nhận diện hành động CSI theo cửa sổ thời gian, không
phải dự đoán ETA hoặc thời gian hoàn thành hành động.

## Nội dung đã chỉnh sửa

### 1. Đồng bộ cấu hình preprocessing

Các hằng số dùng chung được đặt trong `csi_preprocessing.py`:

- Tần số mẫu kỳ vọng: `10 Hz`.
- Kích thước cửa sổ: `30 frame`.
- Bộ lọc Butterworth thông thấp: bậc `4`, cutoff `3 Hz`.
- Chuẩn hóa Z-score theo từng cửa sổ.

`dataset_pipeline.py` và `realtime_predict.py` dùng cùng các hằng số này.
Dataset cũng lưu chính xác danh sách index subcarrier được giữ lại. Realtime
không tự tạo lại mask mà đọc danh sách đó từ checkpoint, tránh trường hợp
model được train trên layout subcarrier khác với layout lúc suy luận.

Nếu các file ghi dữ liệu có mask subcarrier khác nhau, quá trình tạo dataset
dừng với lỗi rõ ràng thay vì ghép sai cột.

### 2. Lưu metadata cùng checkpoint

Checkpoint `csi_cnn_model.pth` hiện có dạng:

```python
{
    "model_state_dict": ...,
    "metadata": {
        "window_size": 30,
        "sample_rate_hz": 20.0,
        "filter_cutoff_hz": 3.0,
        "filter_order": 4,
        "subcarrier_indices": [...],
        "labels": {...},
        "num_classes": 4
    }
}
```

Realtime kiểm tra các metadata bắt buộc trước khi chạy. Checkpoint cũ chỉ chứa
`state_dict` sẽ bị từ chối để tránh chạy với cấu hình không xác định. Cần tạo
lại dataset và train lại model sau khi cập nhật code.

### 3. Đánh giá theo phiên thu, tránh rò rỉ dữ liệu

Trước đây các cửa sổ chồng lấn được chia ngẫu nhiên vào train/test. Vì hai
cửa sổ kế tiếp có thể chứa phần lớn frame giống nhau, accuracy khi đó có thể
cao giả tạo.

Hiện tại:

- Mỗi cửa sổ lưu tên file ghi tương ứng trong `groups`.
- Dữ liệu được chia theo file ghi bằng `GroupShuffleSplit`.
- Có ba nhóm độc lập: train, validation và test.
- Validation dùng để chọn checkpoint và điều chỉnh learning rate.
- Test chỉ được dùng để đánh giá cuối cùng.

Vì vậy, cần có ít nhất ba file ghi độc lập. Tốt hơn là có nhiều file cho mỗi
nhãn, được thu ở các phiên, vị trí hoặc người khác nhau.

### 4. Ổn định kết quả realtime

Realtime hiện:

- Lọc và chuẩn hóa cửa sổ trước khi đưa vào CNN.
- Trung bình xác suất của 5 lần dự đoán gần nhất.
- Hiển thị `Chưa xác định` nếu xác suất cao nhất sau smoothing thấp hơn 60%.
- Theo dõi tốc độ frame thực tế.
- Cảnh báo nếu tốc độ đo được lệch khỏi khoảng `8–12 Hz`.

Giá trị softmax được hiển thị là độ tin cậy tương đối của model, không phải
xác suất model chắc chắn đúng. Muốn diễn giải thành xác suất thực tế cần
calibration riêng trên validation set.

## Quy trình chạy lại sau khi cập nhật

Từ thư mục `tools`:

```powershell
python dataset_pipeline.py
python train.py
python realtime_predict.py --port COM8 --baudrate 2000000
```

Không được bỏ qua bước tạo dataset và train lại nếu đang sử dụng checkpoint
cũ. Khi chạy qua giao diện web, thứ tự vẫn là tạo dataset, train, sau đó
realtime prediction.

## Giới hạn còn lại

Bộ lọc offline hiện vẫn sử dụng `filtfilt` trên toàn bộ file trong lúc tạo
dataset và trên cửa sổ 30 frame trong realtime. Cách này đã được kiểm tra
shape, cutoff và chuẩn hóa để nhất quán cấu hình, nhưng điều kiện biên của
`filtfilt` vẫn có thể khác nhau. Nếu cần độ chính xác cao hơn nữa, bước tiếp
theo là chuyển cả train và realtime sang cùng một bộ lọc causal có state hoặc
resample stream về đúng 20 Hz trước khi lọc.

Độ chính xác cuối cùng cần được đo lại trên các file thu mới, không chỉ dựa
trên accuracy của các cửa sổ được sinh từ cùng phiên thu.
