# Dùng điện thoại với CSI Action Lab

1. Kết nối ESP thu CSI với máy tính bằng USB như trước.
2. Chạy `tools\start_web.bat`. Server mặc định mở cổng 8765 cho mạng LAN.
3. Cho điện thoại và máy tính vào cùng mạng Wi-Fi của router.
4. Mở địa chỉ `Phone / same Wi-Fi` được in trong terminal, hoặc địa chỉ trong khung đầu trang web. Ví dụ: `http://192.168.1.9:8765`.

IP máy tính có thể thay đổi khi đổi mạng hoặc khởi động lại router. Địa chỉ `127.0.0.1` chỉ dùng trên chính máy tính. Điện thoại không cần nối vào AP của ESP nếu máy tính đang dùng Wi-Fi router khác.

Điện thoại có thể thu dữ liệu với countdown, dừng tác vụ, chạy hai phase dự đoán, xem xác suất và xem đồ thị/heatmap CSI. Trạng thái được cập nhật mỗi giây; đồ thị trực tiếp được cập nhật khoảng 300 ms khi đang xem nguồn live. Dữ liệu và model được xử lý, lưu trên máy tính.

Các trình duyệt dùng chung tác vụ trên server. Khi một thiết bị đang thu hoặc dự đoán, thiết bị khác xem được trạng thái và có thể dừng tác vụ đó. Máy tính phải tiếp tục chạy và không vào sleep.

## Nếu chưa truy cập được

- Dừng server cũ rồi chạy lại file batch để áp dụng thay đổi.
- Kiểm tra đúng IP Wi-Fi của máy tính bằng `ipconfig` và đúng cổng in trong terminal.
- Nếu Windows Firewall chặn kết nối, chỉ cho phép Python hoặc TCP 8765 trên mạng Private đang sử dụng; không cần tắt firewall.
- Mạng khách/Guest Wi-Fi có thể chặn các thiết bị truy cập lẫn nhau. Dùng mạng nội bộ không bật client isolation.
- Nếu cổng bận: `tools\start_web.bat --port 8767`, sau đó dùng địa chỉ có cổng 8767.

Server dành cho mạng nội bộ tin cậy, chưa có đăng nhập. Không mở port forwarding ra Internet.

Chỉ mở trên máy tính khi cần: `tools\venv\Scripts\python.exe tools\run.py web --host 127.0.0.1`.
