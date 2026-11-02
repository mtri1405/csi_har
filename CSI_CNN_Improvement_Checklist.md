# Checklist cải thiện hệ thống nhận diện hoạt động người bằng Wi-Fi CSI và CNN

> **Mục tiêu:** rà soát toàn bộ pipeline từ thu thập CSI → tiền xử lý →
> tạo dataset → huấn luyện CNN → đánh giá → dự đoán thời gian thực; tìm
> nguyên nhân khiến kết quả dự đoán bị dao động hoặc thiếu chính xác;
> đồng thời chuẩn bị đầy đủ hình ảnh, biểu đồ và bảng số liệu cho báo
> cáo đồ án.

------------------------------------------------------------------------

## 1. Tổng quan quy trình cần rà soát

``` text
Wi-Fi CSI acquisition
        ↓
Raw CSI inspection & quality check
        ↓
Preprocessing / filtering / normalization
        ↓
Windowing & label assignment
        ↓
Dataset split (train / validation / test)
        ↓
CNN training & hyperparameter tuning
        ↓
Offline evaluation
        ↓
Real-time inference
        ↓
Temporal smoothing / decision logic
        ↓
Real-time activity prediction
```

**Nguyên tắc quan trọng:** không nên vội tăng độ phức tạp của CNN khi
chưa xác nhận dữ liệu CSI, nhãn, cách chia tập dữ liệu và pipeline tiền
xử lý là đúng. Model không thể khắc phục triệt để dữ liệu nhiễu, nhãn
sai hoặc train/test bị chia không hợp lý.

------------------------------------------------------------------------

# PHẦN A --- KIỂM TRA VÀ CẢI THIỆN DỮ LIỆU CSI

## 2. Kiểm tra chất lượng tín hiệu CSI thô

-   [x] Xác định chính xác cấu trúc dữ liệu CSI đang thu:
    -   [x] Số lượng subcarrier.
    -   [ ] Số lượng antenna / spatial stream (nếu có).
    -   [x] CSI amplitude, phase hay cả hai.
    -   [x] Sampling rate thực tế (Hz).
    -   [x] Timestamp của từng packet.
    -   [x] RSSI, packet sequence number và các metadata khác (nếu có).
-   [x] Kiểm tra packet bị mất, timestamp bị trùng hoặc không tăng đều.
-   [x] Kiểm tra khoảng thời gian giữa các packet và mức độ dao động của
    sampling rate.
-   [x] Kiểm tra giá trị CSI bất thường: NaN, Inf, giá trị bằng 0 kéo
    dài hoặc spike đột ngột.
-   [x] Kiểm tra dữ liệu có bị gián đoạn khi thiết bị gửi/nhận hoạt động
    hay không.
-   [x] Thu dữ liệu trong nhiều lần chạy độc lập để đánh giá tính ổn
    định.
-   [ ] Ghi lại điều kiện môi trường khi thu: vị trí thiết bị, khoảng
    cách, hướng đặt antenna, vật cản, số người và hoạt động xung quanh.

### Biểu đồ/hình ảnh nên có

1.  **Raw CSI amplitude theo thời gian**
    -   Trục X: thời gian hoặc packet index.
    -   Trục Y: amplitude.
    -   Hiển thị một số subcarrier đại diện hoặc toàn bộ subcarrier dưới
        dạng heatmap.
2.  **CSI amplitude heatmap**
    -   Trục X: packet/time index.
    -   Trục Y: subcarrier.
    -   Màu sắc: amplitude.
3.  **Inter-packet interval**
    -   Trục X: packet index.
    -   Trục Y: thời gian giữa hai packet liên tiếp (ms).
4.  **Bảng thống kê chất lượng dữ liệu**
    -   Tổng số packet.
    -   Số packet lỗi/mất (nếu xác định được).
    -   Sampling rate trung bình.
    -   Độ lệch chuẩn của inter-packet interval.
    -   Tỷ lệ dữ liệu không hợp lệ.

**Cần trả lời trong báo cáo:** tín hiệu CSI thu được có ổn định không?
Những yếu tố nào có thể làm CSI thay đổi ngoài hoạt động của con người?

------------------------------------------------------------------------

## 3. Kiểm tra dataset và nhãn dữ liệu

-   [x] Liệt kê toàn bộ class hoạt động, ví dụ: `walking`, `sitting`,
    `standing`, `falling` (điều chỉnh theo phạm vi đồ án).
-   [x] Kiểm tra số lượng sample của từng class.
-   [x] Kiểm tra class có bị mất cân bằng hay không.
-   [x] Kiểm tra nhãn có khớp với thời điểm thực hiện hoạt động không.
-   [x] Loại bỏ hoặc đánh dấu những đoạn thu có nhãn không chắc chắn.
-   [x] Kiểm tra có đoạn chuyển tiếp giữa hai hoạt động bị gán nhầm vào
    một class hay không.
-   [x] Đảm bảo mỗi sample/window có đúng một nhãn theo quy tắc đã định.
-   [x] Lưu metadata cho từng phiên thu: session ID, người thực hiện, vị
    trí, ngày thu, thiết bị và class.
-   [x] Tách dữ liệu theo **phiên thu (session)** hoặc người thực hiện
    khi phù hợp, thay vì chia ngẫu nhiên từng window từ cùng một phiên
    vào cả train và test.
-   [ ] Giữ một test set độc lập, không sử dụng để điều chỉnh
    hyperparameter.

### Bảng thống kê dataset

  Class         Số phiên thu   Số window   Tỷ lệ (%) Ghi chú
  ----------- -------------- ----------- ----------- ---------
  Walking                                            
  Sitting                                            
  Standing                                           
  Falling                                            
  **Total**                                      100 

### Biểu đồ/hình ảnh nên có

-   [ ] Bar chart thể hiện số lượng sample/window của từng class.
-   [ ] Timeline minh họa một phiên thu và các đoạn được gán nhãn.
-   [ ] Sơ đồ vị trí bố trí thiết bị phát/thu và khu vực thực hiện hoạt
    động.
-   [ ] Ảnh minh họa quá trình thu dữ liệu thực tế.
-   [ ] Bảng mô tả cấu hình và điều kiện thu dữ liệu.

**Lưu ý về data leakage:** nếu các window chồng lấn được tạo từ cùng một
đoạn CSI, không được để window gần như giống nhau xuất hiện đồng thời ở
train và test. Điều này có thể làm kết quả đánh giá cao giả tạo.

------------------------------------------------------------------------

# PHẦN B --- CẢI THIỆN TIỀN XỬ LÝ CSI

## 4. Rà soát preprocessing pipeline

-   [ ] Vẽ rõ pipeline tiền xử lý hiện tại từ raw CSI đến input của CNN.
-   [ ] Xác định rõ đang sử dụng amplitude, phase hay đặc trưng kết hợp.
-   [ ] Nếu sử dụng phase, kiểm tra bước unwrap và loại bỏ thành phần
    nhiễu/offset phù hợp với phần cứng.
-   [ ] Kiểm tra việc loại bỏ subcarrier không hợp lệ (nếu có).
-   [ ] Kiểm tra filtering có làm mất đặc trưng chuyển động hay không.
-   [ ] Thử so sánh dữ liệu trước và sau filtering.
-   [ ] Kiểm tra normalization:
    -   [ ] Chuẩn hóa theo từng sample hay toàn bộ dataset?
    -   [ ] Thống kê mean/std có được tính từ train set בלבד không?
    -   [ ] Có xử lý trường hợp độ lệch chuẩn bằng 0 không?
-   [ ] Kiểm tra window size và stride.
-   [ ] Thử một số window size khác nhau dựa trên sampling rate và thời
    lượng hoạt động.
-   [ ] Kiểm tra cách xử lý window thiếu packet hoặc có độ dài không
    đồng nhất.
-   [ ] Đảm bảo preprocessing lúc inference giống hệt preprocessing lúc
    training.

### Những thông số cần ghi lại

  Thông số                Giá trị hiện tại   Giá trị thử nghiệm   Ghi chú
  ----------------------- ------------------ -------------------- -------------------
  Sampling rate                                                   Hz
  Số subcarrier sử dụng                                           
  CSI feature                                                     Amplitude/Phase
  Filter                                                          Loại và tham số
  Window size                                                     Samples hoặc giây
  Window stride                                                   Samples hoặc giây
  Normalization                                                   
  Input shape                                                     Ví dụ: C × T

### Biểu đồ/hình ảnh nên có

1.  CSI raw trước và sau filtering trên cùng một khoảng thời gian.
2.  CSI amplitude heatmap trước và sau preprocessing.
3.  Minh họa windowing: window size, stride và vùng overlap.
4.  Histogram phân phối giá trị feature trước/sau normalization.
5.  Hình minh họa tensor đầu vào CNN (channel × time hoặc cấu trúc thực
    tế đang sử dụng).

**Cần trả lời trong báo cáo:** mỗi bước tiền xử lý giải quyết vấn đề gì?
Có bằng chứng cho thấy bước đó cải thiện tín hiệu mà không làm mất thông
tin quan trọng không?

------------------------------------------------------------------------

# PHẦN C --- KIỂM TRA VÀ CẢI THIỆN CNN

## 5. Kiểm tra kiến trúc model

-   [ ] Ghi lại chính xác kiến trúc CNN hiện tại.
-   [ ] Liệt kê số layer, kernel size, stride, padding, activation và
    pooling.
-   [ ] Kiểm tra input shape có khớp với dữ liệu sau preprocessing
    không.
-   [ ] Kiểm tra output layer có số neuron bằng số class không.
-   [ ] Kiểm tra loss function có phù hợp với cách encode label không.
-   [ ] Kiểm tra cách chuyển logits thành xác suất (softmax nếu phù
    hợp).
-   [ ] Tính số lượng parameters của model.
-   [ ] Kiểm tra model có quá lớn so với kích thước dataset không.
-   [ ] Kiểm tra model có dấu hiệu underfitting hoặc overfitting.
-   [ ] Lưu cấu hình model và phiên bản code cho từng lần thử nghiệm.

### Bảng mô tả kiến trúc CNN

  Layer           Output shape   Kernel / Pool   Activation         Parameters
  --------------- -------------- --------------- ---------------- ------------
  Input                                          \-                          0
  Conv1D/Conv2D                                  ReLU/...         
  Pooling                                        \-               
  ...                                                             
  Dense                                          ReLU/...         
  Output                                         Softmax/logits   
  **Total**                                                       

-   [ ] Thử nghiệm thay đổi từng yếu tố một để biết yếu tố nào tạo ra
    khác biệt.
-   [ ] Thử điều chỉnh learning rate.
-   [ ] Thử batch size phù hợp với kích thước dataset và tài nguyên tính
    toán.
-   [ ] Sử dụng early stopping dựa trên validation loss.
-   [ ] Lưu checkpoint của model tốt nhất theo validation metric.
-   [ ] Thử regularization như dropout hoặc weight decay nếu có dấu hiệu
    overfitting.
-   [ ] Chỉ dùng class weight hoặc resampling khi dataset thực sự mất
    cân bằng.
-   [ ] Cân nhắc augmentation có cơ sở vật lý và không làm thay đổi nhãn
    hoạt động.

## 6. Theo dõi quá trình training

-   [ ] Lưu training loss theo từng epoch.
-   [ ] Lưu validation loss theo từng epoch.
-   [ ] Lưu training accuracy và validation accuracy.
-   [ ] Lưu learning rate theo epoch nếu có scheduler.
-   [ ] Ghi lại thời gian training.
-   [ ] Ghi lại batch size, epoch, optimizer, learning rate và random
    seed.
-   [ ] Chạy nhiều lần với seed khác nhau nếu có đủ thời gian để đánh
    giá độ ổn định.
-   [ ] Không lựa chọn model dựa trên test accuracy trong quá trình
    tuning.

### Biểu đồ bắt buộc nên có

1.  **Training Loss vs Validation Loss**
    -   Dùng để phát hiện overfitting/underfitting.
2.  **Training Accuracy vs Validation Accuracy**
    -   Dùng để theo dõi khả năng học của model.
3.  **Learning Rate vs Epoch**
    -   Nên có nếu sử dụng learning-rate scheduler.
4.  **Thời gian training**
    -   Có thể trình bày trong bảng hoặc biểu đồ so sánh các cấu hình.

### Cách đọc learning curves

  -----------------------------------------------------------------------
  Hiện tượng              Khả năng nguyên nhân    Hướng kiểm tra
  ----------------------- ----------------------- -----------------------
  Train tốt, validation   Overfitting, dữ liệu    Kiểm tra split, tăng dữ
  kém                     chia không đại diện     liệu, regularization

  Cả train và validation  Underfitting, feature   Kiểm tra preprocessing,
  đều kém                 chưa tốt, nhãn nhiễu    model, label

  Validation loss tăng    Overfitting             Early stopping,
  trong khi train loss                            regularization
  giảm                                            

  Loss dao động mạnh      Learning rate cao,      Kiểm tra learning rate
                          batch nhỏ, dữ liệu      và dữ liệu
                          không ổn định           

  Accuracy cao bất thường Có thể data leakage     Chia lại theo
                          hoặc split sai          session/person
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# PHẦN D --- ĐÁNH GIÁ MODEL MỘT CÁCH ĐÚNG ĐẮN

## 7. Đánh giá offline trên test set

-   [ ] Chỉ đánh giá trên test set độc lập sau khi đã chốt model.
-   [ ] Tính Accuracy.
-   [ ] Tính Precision, Recall và F1-score cho từng class.
-   [ ] Báo cáo Macro-F1 khi các class có số lượng không cân bằng.
-   [ ] Tạo confusion matrix.
-   [ ] Xác định cặp class thường bị nhầm lẫn.
-   [ ] Kiểm tra kết quả theo từng session/người thực hiện nếu dataset
    có metadata.
-   [ ] Báo cáo mean và standard deviation qua nhiều lần chạy nếu có
    thực hiện.
-   [ ] Lưu dự đoán và nhãn thật để có thể phân tích lỗi.

### Biểu đồ/bảng nên có

1.  **Confusion Matrix**
    -   Hàng: nhãn thật.
    -   Cột: nhãn dự đoán.
    -   Nên hiển thị cả số lượng và tỷ lệ chuẩn hóa.
2.  **Precision / Recall / F1-score theo class**
    -   Dạng grouped bar chart.
3.  **Bảng classification report**
    -   Precision, Recall, F1-score, Support.
4.  **Bảng so sánh các phiên bản model**
    -   Accuracy, Macro-F1, số parameters, thời gian inference.

### Bảng kết quả tổng hợp

  -----------------------------------------------------------------------------
  Model version       Accuracy     Macro-F1   Parameters    Inference Ghi chú
                                                                 time 
  --------------- ------------ ------------ ------------ ------------ ---------
  Baseline                                                            

  Preprocessing                                                       
  v2                                                                  

  CNN tuned                                                           

  Final model                                                         
  -----------------------------------------------------------------------------

**Không chỉ nhìn Accuracy:** ví dụ model có thể dự đoán tốt class
`standing` nhưng thường xuyên nhầm `sitting` và `walking`. Confusion
matrix và F1-score theo class sẽ cho thấy vấn đề này.

------------------------------------------------------------------------

# PHẦN E --- CẢI THIỆN DỰ ĐOÁN THỜI GIAN THỰC

## 8. Kiểm tra pipeline real-time

-   [ ] Đảm bảo luồng CSI real-time có cùng cấu trúc và thứ tự feature
    với dữ liệu training.
-   [ ] Dùng đúng filter, normalization, window size và stride như lúc
    training.
-   [ ] Kiểm tra buffer có đủ dữ liệu trước khi inference.
-   [ ] Kiểm tra xử lý packet đến trễ, mất packet hoặc timestamp không
    đều.
-   [ ] Đảm bảo không vô tình dùng dữ liệu tương lai khi tạo window
    real-time.
-   [ ] Đo thời gian từ lúc nhận packet đến lúc tạo ra prediction.
-   [ ] Đo inference latency của model riêng biệt.
-   [ ] Đo tổng end-to-end latency.
-   [ ] Kiểm tra tốc độ cập nhật prediction (prediction rate).
-   [ ] Kiểm tra CPU/RAM khi chạy liên tục.
-   [ ] Kiểm tra model có bị load lại hoặc preprocessing có bị khởi tạo
    lại không cần thiết hay không.

## 9. Giảm hiện tượng prediction bị nhảy liên tục

Trước khi thêm thuật toán làm mượt, cần lưu chuỗi xác suất đầu ra của
model để xác định nguyên nhân là model không chắc chắn hay tín hiệu đầu
vào thay đổi mạnh.

Các hướng thử nghiệm:

-   [ ] **Sliding window:** chỉ dự đoán khi buffer đã có đủ số sample
    cần thiết.
-   [ ] **Overlap giữa các window:** cập nhật prediction theo stride nhỏ
    hơn window size, nhưng cần đánh giá độ trễ và mức độ tương quan giữa
    các prediction.
-   [ ] **Moving average trên xác suất:** lấy trung bình xác suất của
    một số prediction gần nhất rồi chọn class có xác suất cao nhất.
-   [ ] **Majority voting:** chọn class xuất hiện nhiều nhất trong một
    cửa sổ prediction gần đây.
-   [ ] **Confidence threshold:** chỉ công bố class khi confidence vượt
    ngưỡng đã được xác định trên validation set; nếu không, trả về trạng
    thái `uncertain`.
-   [ ] **Hysteresis / minimum duration:** tránh đổi class chỉ vì một
    prediction đơn lẻ; yêu cầu bằng chứng ổn định trong một khoảng thời
    gian.
-   [ ] **Unknown / transition state:** cân nhắc trạng thái không xác
    định hoặc chuyển tiếp khi tín hiệu không đủ rõ.

**Cảnh báo:** smoothing có thể làm prediction nhìn ổn định hơn nhưng
cũng tăng độ trễ và che giấu lỗi của model. Phải đánh giá cả độ chính
xác lẫn latency trước/sau smoothing. Không chọn ngưỡng chỉ dựa trên test
set.

### Biểu đồ nên có

1.  **Prediction timeline**
    -   Trục X: thời gian.
    -   Trục Y: class.
    -   Vẽ ground truth và predicted class trên cùng biểu đồ.
2.  **Class probability theo thời gian**
    -   Mỗi class là một đường xác suất.
3.  **Prediction trước và sau smoothing**
    -   Hai biểu đồ hoặc hai subplot cùng khoảng thời gian.
4.  **Confidence distribution**
    -   Histogram confidence của prediction đúng và sai.
5.  **Latency distribution**
    -   Histogram hoặc boxplot thời gian inference/end-to-end.
6.  **Prediction update interval**
    -   Thời gian giữa hai lần cập nhật kết quả.

## 10. Đánh giá real-time trong nhiều tình huống

-   [ ] Test từng class trong điều kiện môi trường cố định.
-   [ ] Test nhiều lần lặp lại cùng một hoạt động.
-   [ ] Test khi thay đổi vị trí người thực hiện.
-   [ ] Test khi thay đổi khoảng cách hoặc hướng thiết bị trong phạm vi
    đồ án.
-   [ ] Test khi có người khác di chuyển trong môi trường (nếu phù hợp).
-   [ ] Test khi không có hoạt động mục tiêu / môi trường nền.
-   [ ] Test các đoạn chuyển tiếp giữa hai hoạt động.
-   [ ] Ghi video hoặc ảnh chụp màn hình giao diện real-time để minh
    họa.
-   [ ] Lưu log gồm timestamp, ground truth, predicted class và
    confidence.
-   [ ] Lập bảng kết quả theo từng kịch bản.

### Bảng đánh giá real-time

  ----------------------------------------------------------------------------
  Scenario     Ground        Số lần thử Accuracy / F1 Latency trung Nhận xét
               truth                                           bình 
  ------------ ---------- ------------- ------------- ------------- ----------
  Scenario 1   Walking                                              

  Scenario 2   Sitting                                              

  Scenario 3   Standing                                             

  Scenario 4   Falling                                              

  Transition   N/A                                                  

  Background / N/A                                                  
  no target                                                         
  ----------------------------------------------------------------------------

------------------------------------------------------------------------

# PHẦN F --- THIẾT KẾ THỬ NGHIỆM CÓ KIỂM SOÁT

## 11. Thực hiện ablation study

Không nên thay đổi nhiều thành phần cùng lúc. Hãy tạo các phiên bản thử
nghiệm để biết chính xác yếu tố nào giúp cải thiện kết quả.

-   [ ] Baseline: pipeline hiện tại.
-   [ ] Thử amplitude-only so với feature hiện tại.
-   [ ] Thử các phương án filtering.
-   [ ] Thử các window size/stride khác nhau.
-   [ ] Thử normalization khác nhau.
-   [ ] Thử CNN với một số cấu hình có kiểm soát.
-   [ ] Thử smoothing trên cùng một model.
-   [ ] So sánh kết quả trên cùng một cách chia dataset.

### Bảng ablation study

  ---------------------------------------------------------------------------------
  Experiment   Thay đổi duy         Accuracy      Macro-F1     Real-time Kết luận
               nhất                                              latency 
  ------------ --------------- ------------- ------------- ------------- ----------
  E0           Baseline                                                  

  E1           Thay filter                                               

  E2           Thay window                                               
               size                                                      

  E3           Thay                                                      
               normalization                                             

  E4           Điều chỉnh CNN                                            

  E5           Thêm temporal                                             
               smoothing                                                 
  ---------------------------------------------------------------------------------

**Quy tắc:** mỗi experiment nên có ID, cấu hình, seed, dataset version,
model checkpoint và kết quả. Không ghi đè kết quả của lần chạy trước.

------------------------------------------------------------------------

# PHẦN G --- HÌNH ẢNH VÀ BIỂU ĐỒ CHO BÁO CÁO

## 12. Danh sách hình ảnh đề xuất

  -----------------------------------------------------------------------------
                    STT Hình ảnh / biểu đồ    Nội dung cần thể Mức độ ưu tiên
                                              hiện             
  --------------------- --------------------- ---------------- ----------------
                      1 System architecture   Sơ đồ tổng thể   Bắt buộc
                                              phần cứng, thu   
                                              CSI, xử lý, CNN, 
                                              kết quả          

                      2 Hardware setup        Thiết bị,        Bắt buộc
                                              antenna, vị trí  
                                              phát/thu, khu    
                                              vực thử nghiệm   

                      3 Data collection       Quy trình thu,   Nên có
                        workflow              gán nhãn và lưu  
                                              dữ liệu          

                      4 Raw CSI plot          Tín hiệu CSI thô Bắt buộc
                                              theo thời gian   

                      5 CSI heatmap           Phân bố CSI theo Nên có
                                              subcarrier và    
                                              thời gian        

                      6 Preprocessing         Các bước lọc,    Bắt buộc
                        pipeline              chuẩn hóa,       
                                              windowing        

                      7 Before/after          So sánh tín hiệu Nên có
                        filtering             trước và sau lọc 

                      8 Dataset distribution  Số lượng mẫu của Bắt buộc
                                              từng class       

                      9 CNN architecture      Các layer và     Bắt buộc
                                              kích thước       
                                              tensor           

                     10 Training/validation   Quá trình hội tụ Bắt buộc
                        loss                  của model        

                     11 Training/validation   Khả năng học qua Bắt buộc
                        accuracy              từng epoch       

                     12 Confusion matrix      Các class bị     Bắt buộc
                                              nhầm lẫn         

                     13 Per-class metrics     Precision,       Bắt buộc
                                              Recall, F1-score 

                     14 Real-time prediction  Ground truth và  Bắt buộc
                        timeline              prediction theo  
                                              thời gian        

                     15 Probability timeline  Confidence của   Nên có
                                              từng class       

                     16 Latency distribution  Độ trễ inference Nên có
                                              và end-to-end    

                     17 Before/after          Tác động của bộ  Nên có nếu sử
                        smoothing             làm mượt         dụng

                     18 Experiment comparison So sánh baseline Bắt buộc
                                              và các phiên bản 
                                              cải tiến         

                     19 Real-time application Giao diện dự     Nên có
                        screenshot            đoán thực tế     

                     20 Failure case examples Một số tình      Nên có
                                              huống model dự   
                                              đoán sai và phân 
                                              tích             
  -----------------------------------------------------------------------------

## 13. Danh sách bảng số liệu đề xuất

-   [ ] Bảng thông số phần cứng và cấu hình Wi-Fi.
-   [ ] Bảng thông số CSI và sampling rate.
-   [ ] Bảng mô tả các class và quy trình gán nhãn.
-   [ ] Bảng phân chia train/validation/test theo session hoặc subject.
-   [ ] Bảng cấu hình preprocessing.
-   [ ] Bảng kiến trúc CNN và số parameters.
-   [ ] Bảng hyperparameters training.
-   [ ] Bảng kết quả classification report.
-   [ ] Bảng so sánh các experiment/ablation study.
-   [ ] Bảng latency, prediction rate và tài nguyên sử dụng.
-   [ ] Bảng kết quả đánh giá real-time theo từng scenario.

------------------------------------------------------------------------

# PHẦN H --- CẤU TRÚC NỘI DUNG ĐƯA VÀO BÁO CÁO

## 14. Gợi ý bố cục chương triển khai và đánh giá

### Chương 1 --- Giới thiệu đề tài

-   Bài toán nhận diện hoạt động bằng Wi-Fi CSI.
-   Mục tiêu và phạm vi.
-   Các hoạt động cần nhận diện.
-   Yêu cầu hệ thống và tiêu chí đánh giá.

### Chương 2 --- Cơ sở lý thuyết

-   Wi-Fi CSI là gì.
-   CSI amplitude và phase.
-   Ảnh hưởng của chuyển động lên kênh truyền.
-   Tổng quan CNN và bài toán classification.
-   Các chỉ số đánh giá model.

### Chương 3 --- Thiết kế và xây dựng hệ thống

-   Sơ đồ kiến trúc tổng thể.
-   Thiết bị và môi trường thử nghiệm.
-   Quy trình thu thập CSI.
-   Cấu trúc dữ liệu và gán nhãn.
-   Pipeline preprocessing.
-   Thiết kế CNN.
-   Pipeline dự đoán thời gian thực.

### Chương 4 --- Thực nghiệm và đánh giá

-   Cấu hình phần cứng/phần mềm.
-   Dataset và cách chia tập.
-   Cấu hình training.
-   Learning curves.
-   Confusion matrix và classification report.
-   Ablation study.
-   Real-time prediction timeline.
-   Đánh giá latency và độ ổn định.
-   Phân tích các trường hợp dự đoán sai.

### Chương 5 --- Kết luận và hướng phát triển

-   Những phần đã hoàn thành.
-   Kết quả đạt được so với mục tiêu.
-   Hạn chế: dữ liệu, môi trường, số người tham gia, độ trễ...
-   Hướng phát triển: tăng dữ liệu, đánh giá cross-session/cross-person,
    tối ưu inference và mở rộng class.

------------------------------------------------------------------------

# PHẦN I --- KẾ HOẠCH THỰC HIỆN THEO THỨ TỰ ƯU TIÊN

## 15. Checklist triển khai

### Giai đoạn 1: Chẩn đoán lỗi hiện tại

-   [ ] Chạy lại pipeline hiện tại và lưu toàn bộ log.
-   [ ] Vẽ raw CSI plot và CSI heatmap.
-   [ ] Kiểm tra packet loss, timestamp và sampling rate.
-   [ ] Kiểm tra label và số lượng sample mỗi class.
-   [ ] Kiểm tra cách chia train/validation/test để phát hiện data
    leakage.
-   [ ] Vẽ learning curves và confusion matrix của model hiện tại.
-   [ ] Lưu chuỗi prediction và confidence khi chạy real-time.

**Đầu ra cần có:** xác định được vấn đề chính nằm ở dữ liệu,
preprocessing, model hay real-time pipeline.

### Giai đoạn 2: Chuẩn hóa dữ liệu và preprocessing

-   [ ] Chốt format dữ liệu CSI.
-   [ ] Chốt cách gán nhãn và metadata.
-   [ ] Chốt cách chia dataset theo session/person.
-   [ ] Kiểm tra và chốt preprocessing.
-   [ ] Thử window size/stride có kiểm soát.
-   [ ] Đảm bảo train và inference dùng chung logic preprocessing.

**Đầu ra cần có:** dataset version rõ ràng và pipeline preprocessing có
thể tái lập.

### Giai đoạn 3: Cải thiện và đánh giá CNN

-   [ ] Thiết lập baseline.
-   [ ] Điều chỉnh learning rate, batch size và regularization.
-   [ ] Thử nghiệm kiến trúc CNN có kiểm soát.
-   [ ] Dùng validation set để chọn model.
-   [ ] Chốt model và chỉ đánh giá test set sau cùng.
-   [ ] Xuất classification report, confusion matrix và learning curves.

**Đầu ra cần có:** model cuối cùng cùng các chỉ số đánh giá có thể kiểm
chứng.

### Giai đoạn 4: Ổn định real-time inference

-   [ ] Đồng bộ preprocessing giữa training và real-time.
-   [ ] Kiểm tra buffer, windowing và timestamp.
-   [ ] Đo inference latency và end-to-end latency.
-   [ ] Phân tích prediction timeline và confidence.
-   [ ] Thử smoothing/majority voting/confidence threshold.
-   [ ] Đo lại độ chính xác và latency sau khi smoothing.
-   [ ] Test nhiều kịch bản và lưu log.

**Đầu ra cần có:** demo real-time ổn định hơn và số liệu đánh giá thực
tế.

### Giai đoạn 5: Hoàn thiện báo cáo

-   [ ] Vẽ system architecture.
-   [ ] Chụp hardware setup và giao diện real-time.
-   [ ] Xuất biểu đồ CSI raw/preprocessing.
-   [ ] Xuất dataset distribution.
-   [ ] Vẽ CNN architecture.
-   [ ] Xuất learning curves.
-   [ ] Xuất confusion matrix và classification report.
-   [ ] Xuất real-time prediction timeline.
-   [ ] Xuất bảng latency và experiment comparison.
-   [ ] Viết phân tích failure cases và limitations.
-   [ ] Đảm bảo mọi hình/bảng có số thứ tự, tiêu đề, đơn vị và được nhắc
    đến trong nội dung báo cáo.

------------------------------------------------------------------------

# PHẦN J --- CÁC FILE LOG NÊN LƯU CHO MỖI LẦN THỬ

## 16. Gợi ý tổ chức thư mục

``` text
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── preprocessing/
├── models/
│   ├── checkpoints/
│   └── model_configs/
├── experiments/
│   ├── E00_baseline/
│   │   ├── config.yaml
│   │   ├── train_log.csv
│   │   ├── metrics.json
│   │   ├── predictions.csv
│   │   └── notes.md
│   ├── E01_filtering/
│   └── E02_window_size/
├── realtime_logs/
├── figures/
│   ├── csi/
│   ├── dataset/
│   ├── training/
│   ├── evaluation/
│   └── realtime/
└── report/
```

Mỗi experiment nên lưu tối thiểu:

-   Dataset version và cách chia train/validation/test.
-   Preprocessing configuration.
-   Model architecture/configuration.
-   Hyperparameters và random seed.
-   Training/validation loss và accuracy.
-   Best checkpoint.
-   Test metrics (chỉ sau khi chốt model).
-   Real-time prediction log và latency log.
-   Ghi chú về thay đổi so với experiment trước.

------------------------------------------------------------------------

# PHẦN K --- TIÊU CHÍ HOÀN THÀNH

## 17. Definition of Done

Chỉ nên xem hệ thống đã được đánh giá tương đối đầy đủ khi có thể trả
lời được các câu hỏi sau:

-   [ ] Dữ liệu CSI có ổn định và có thể tái lập quá trình thu không?
-   [ ] Dataset có đủ đại diện cho từng class không?
-   [ ] Nhãn và cách chia dataset có đáng tin cậy không?
-   [ ] Preprocessing lúc training và inference có giống nhau không?
-   [ ] Model có tổng quát hóa trên session/test set độc lập không?
-   [ ] Có biết chính xác class nào đang bị nhầm nhiều nhất không?
-   [ ] Có phân tích được nguyên nhân của các prediction sai không?
-   [ ] Có đo được inference latency và end-to-end latency không?
-   [ ] Việc smoothing có cải thiện độ ổn định mà không làm latency tăng
    quá mức không?
-   [ ] Kết quả real-time có được kiểm tra qua nhiều lần chạy và nhiều
    kịch bản không?
-   [ ] Báo cáo có đủ hình ảnh, biểu đồ, bảng số liệu và mô tả phương
    pháp để người khác hiểu và tái lập không?

------------------------------------------------------------------------

## 18. Khuyến nghị: nên làm gì ngay bây giờ?

Với tình trạng **đã có code thu CSI, preprocessing, CNN training và
real-time inference nhưng prediction còn bị loạn**, nên ưu tiên theo thứ
tự sau:

1.  **Đừng thay CNN ngay.** Giữ model hiện tại làm baseline.
2.  **Lưu một phiên real-time có ground truth**, đồng thời log CSI
    input, timestamp, prediction và confidence.
3.  **Vẽ CSI raw/heatmap và prediction timeline.** Xác định tín hiệu đầu
    vào có nhiễu/gián đoạn hay model thực sự phân loại không ổn định.
4.  **Kiểm tra data split và label.** Đặc biệt kiểm tra việc window
    chồng lấn hoặc cùng session xuất hiện ở cả train và test.
5.  **Kiểm tra preprocessing train/inference có giống nhau tuyệt đối
    không.**
6.  **Vẽ learning curves và confusion matrix** để xác định model đang
    overfit, underfit hay nhầm một số class cụ thể.
7.  Sau khi xử lý các vấn đề trên, mới thử điều chỉnh window size,
    filtering, normalization và CNN.
8.  Cuối cùng mới thử smoothing/majority voting và đo lại latency.

**Thứ tự ưu tiên cốt lõi:** Data quality → Label & split → Preprocessing
consistency → Model evaluation → Real-time pipeline → Temporal
smoothing.

Đây là cách tiếp cận giúp cải thiện hệ thống dựa trên bằng chứng, thay
vì thay đổi nhiều tham số cùng lúc và không biết yếu tố nào thực sự có
tác dụng.
