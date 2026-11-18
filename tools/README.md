# Công cụ CSI

Truy cập bằng điện thoại cùng Wi-Fi: chạy `start_web.bat`, mở địa chỉ **Phone / same Wi-Fi** in trong terminal. Xem [hướng dẫn điện thoại](MOBILE_ACCESS.md).

Mã đang dùng nằm trong `csi/`, giao diện trong `web/`, kiểm thử trong `tests/`. Dữ liệu CSV giữ tại `Data/`, đầu ra giữ tại `reports/`.

Từ thư mục gốc dự án:

```powershell
tools/venv/Scripts/python.exe -m pip install -r tools/requirements.txt
tools/venv/Scripts/python.exe -X utf8 tools/run.py web
```

Hoặc chạy `start_web.bat`; mở http://127.0.0.1:8765.

Một điểm chạy chung: `run.py`. Các lệnh hỗ trợ: `web`, `ports`, `record`, `prepare`, `plot`, `train`, `predict`, `monitor`, `audit`. Ví dụ `tools/venv/Scripts/python.exe tools/run.py record --help`.

Kiểm thử:

```powershell
tools/venv/Scripts/python.exe -m unittest discover -s tools/tests -p test_*.py
```

Xem [README dự án](../README.md) để biết kiến trúc, cách thu dữ liệu, huấn luyện và các giới hạn. Nếu web đang chạy phiên cũ, hãy dừng và mở lại sau khi cập nhật cấu trúc.
