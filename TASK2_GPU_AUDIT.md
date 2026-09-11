# Đối chiếu thuật toán Task 2: NumPy và bản Torch

Ngày tổng hợp: 2026-09-11.

## Kết luận

- Không tìm thấy việc thay công thức hoặc dùng thuật toán xấp xỉ mới để tăng
  tốc độ trong bản Torch, so với **Task 2 CPU có bật RFF** trong repo.
- Hai backend không luôn cho kết quả số hoặc nhãn giống hệt nhau: phép thử
  thực tế đã tìm thấy sai lệch số học và một phản ví dụ đổi nhãn.
- Reuse projection khớp chính xác với fit lại trên backend Torch trong 15 tổ
  hợp đã thử; tối ưu này không phải nguyên nhân của sai lệch NumPy–Torch.
- **Chưa kiểm chứng CUDA thực tế.** Máy kiểm tra chỉ có Torch bản CPU; không
  suy rộng kết quả này thành chứng nhận tương đương trên GPU/T4 của VM.
- Không sửa code thuật toán, cách xử lý hòa hoặc ngưỡng novelty trong audit.
  Tolerance dưới đây chỉ dùng để đo sai lệch, không áp vào dự đoán.

## Baseline và môi trường

Baseline: `examples/lim-models-mnd-rff-ref-lim.py`, dùng `limnfst/models.py`,
`limnfst/nfst.py`, `limnfst/mapping.py`, `limnfst/novelty.py`.
Bản được kiểm tra: `examples/lim-models-gpu.py`.

RFF vốn là bước xấp xỉ kernel có sẵn trong baseline, không phải thủ thuật mới
của GPU. Task 1 mặc định không bật RFF là cấu hình thuật toán khác. Audit này
đối chiếu implementation trong repo, không tái thẩm định độc lập toàn bộ
implementation theo bài báo gốc.

| Thành phần | Phiên bản / thiết lập thực chạy |
| --- | --- |
| NumPy / pandas / scikit-learn | 2.4.6 / 3.0.5 / 1.9.1 |
| PyTorch | 2.14.0+cpu |
| Device / CUDA khả dụng | `cpu` / `False` |
| Dtype các tensor model đã quan sát | `float64` |
| Luồng tính toán | Torch và thư viện số giới hạn 1 luồng |
| So sánh mảng | `np.allclose(..., rtol=1e-7, atol=1e-9)` |
| So sánh nhãn / reuse | So sánh chính xác, không dùng tolerance |

## Đối chiếu code

| Công đoạn | Kết quả rà soát |
| --- | --- |
| Chia dữ liệu và preprocessing | Cùng nested LOCO, scaler và seed; mảng preprocessing khớp chính xác trên 5 fold StandardScaler đã thử |
| RFF | Cùng công thức gamma, sklearn `RBFSampler`, seed, random weights và offsets; Torch tính phép biến đổi từ các tham số đó |
| Fit/reference split | Cùng phép chia stratified; CPU chia mảng, Torch chia các chỉ số tương ứng |
| Chuẩn hóa | Cùng centering theo từng hàng và chuẩn L2 |
| Học phép chiếu | Cùng regularized solve, residual được centering, reduced QR, pseudoinverse, between-class eigendecomposition và simplex alignment |
| Pseudoinverse | Torch đặt `atol=0.0, rtol=1e-15` để khớp cutoff mặc định của NumPy trong baseline |
| Khoảng cách | Tính đầy đủ cùng tập khoảng cách bình phương; `topk` không phải approximate nearest neighbors |
| Batching | Chia hàng query để xử lý, không bỏ bớt reference points |
| Ngưỡng novelty | Cùng leave-one-out, giới hạn k theo số điểm khả dụng, quantile tuyến tính và sàn `1e-12` |
| Quyết định nhãn | Cùng `argmin` và quy tắc loại novelty nghiêm ngặt `> 1` |
| Grid | Chỉ gộp cấu hình khác `neighbors` và `novelty_quantile`; mọi tham số khác vẫn thuộc khóa nhóm |

Không thấy AMP, hạ model xuống float16/float32, approximate nearest neighbors
hoặc lấy mẫu bớt dữ liệu train ngoài phép chia fit/reference vốn có.

17 hàm CPU/Torch có AST giống nhau: `split_outer_data`, `split_outer_mnd_data`,
`sanitize_features`, `preprocess_mnd_split`, `prepare_inner_loco_folds`,
`prepare_outer_class_grid`, `prepare_grid_fold_cache`, `make_model`,
`calculate_metrics`, `evaluate_model`, `evaluate_configuration`,
`grid_configurations`, `configuration_name`, `configuration_id`,
`summarize_grid_runs`, `candidate_from_runs`, `select_best_parameters`.
AST giống nhau không chứng minh hai thư viện số cho kết quả bitwise giống nhau.

## Phép thử thực tế

BoT_IoT được nạp có 10.118 hàng. Thử seed 42, outer novel class `0Normal`,
inner novel class `HTTP`, fold 1 trong 5 fold. Fold StandardScaler có 4.391
mẫu train, 2.739 mẫu query, 22 features.

Đã thử **225 nhóm tham số phép chiếu**: 5 scaler × 5 reference size × 3 RFF
size × 3 gamma multiplier; giữ `k=5`, quantile `0.95`, epsilon `1e-4`.
Đây **không phải** chạy đủ mọi outer/inner class, seed, fold và cả 3.375 cấu
hình của thí nghiệm đầy đủ. 616.275 là số cặp mẫu–cấu hình, không phải số mẫu
độc lập.

| Bộ kiểm tra | Số ca | Cặp mẫu–cấu hình | Nhãn closed khác | Nhãn open khác | Ca có mảng không đạt allclose |
| --- | ---: | ---: | ---: | ---: | ---: |
| BoT_IoT | 225 | 616.275 | 0 | 0 | **210** |
| Synthetic, nhiều seed, 2/3/5 lớp, bật/tắt RFF | 12 | 960 | 0 | 0 | 0 |
| Dữ liệu suy biến | 3 | 225 | **5** | **5** | 0 |
| Stress, epsilon `1e-8` / `1e-12` | 2 | 150 | 0 | 0 | **2** |

Không có exception. Không được diễn giải thành "mọi kiểm tra đều pass":
BoT_IoT có sai lệch mảng vượt tolerance, còn dữ liệu suy biến có nhãn khác
ngay cả khi các mảng vẫn đạt tolerance.

### Sai lệch trên BoT_IoT

| Đại lượng | Sai lệch tuyệt đối lớn nhất | Sai lệch chuẩn tương đối lớn nhất |
| --- | ---: | ---: |
| `theta_` | `6.6470693e-6` | `2.0183685e-10` |
| `projection_matrix_` | `4.3798832e-7` | `2.1489299e-9` |
| `reference_thresholds_` | `2.3643741e-5` | `6.9079227e-6` |
| `reference_scores` | `1.2249823e-3` | `3.5121546e-8` |
| `novelty_scores` | **`3.7390316`** | `4.0020260e-6` |

Chuẩn tương đối là `norm(NumPy - Torch) / max(norm(NumPy), 1e-30)`.
Hai cột lấy cực đại riêng trên 225 ca, không nhất thiết từ cùng một ca.
Chuẩn tương đối nhỏ không đảm bảo mọi phần tử đạt allclose.

Ví dụ `RobustScaler, ref=0.3, rff=512, gamma_multiplier=1.0`: novelty của một
query là `926423.0294483535` ở NumPy và `926426.7684799917` ở Torch.
Sai lệch tương đối tại query đó khoảng `4.036e-6`; cả hai vẫn lớn hơn ngưỡng 1
nên không đổi quyết định. Không thể bảo đảm query gần ngưỡng cũng sẽ giữ nhãn.

### Phản ví dụ đổi nhãn

Synthetic 3 features, 3 lớp, seed dữ liệu 99: 40 mẫu train/lớp, 15 query/lớp
và 30 query bổ sung. Đặt hàng train đầu tiên bằng 7 và cột cuối trùng cột đầu
ở train/query. Model dùng seed 42, reference size 0.25, epsilon `1e-4`, không RFF.

**5/75 query đổi nhãn lớp**, từ lớp 0 ở NumPy sang lớp 1 ở Torch; không có
thay đổi cờ loại novelty. Các nhãn open khác do kế thừa nhãn lớp.
Ví dụ hai khoảng cách nhỏ nhất của query 1:

- NumPy: `[8.456036162075207e-34, 8.456036162075207e-34]`.
- Torch: `[8.58242531467562e-34, 8.614022602825723e-34]`.

Bằng chứng phù hợp với độ nhạy số học khi khoảng cách hòa/gần 0, không cho
thấy shortcut tìm láng giềng. Không thêm epsilon vào `argmin` hoặc ép nhãn
theo NumPy: làm như vậy sẽ tự thay đổi quy tắc của baseline.

Ở hai ca BoT_IoT có sai lệch novelty lớn, truyền **cùng mảng đã chuẩn hóa** vào
hai bộ giải vẫn tạo projection khác nhau: max abs khoảng `6.58e-8` và `1.36e-7`.
Khác biệt không chỉ đến từ bước tính RFF mà còn từ đại số tuyến tính giữa backend.

## Kiểm tra primitive, reuse và resume

- 336 kiểm tra distance/threshold đều đạt tolerance: batch size 1/3/7/256,
  reference count 2/9/17, k 1/3/9/50, điểm trùng, leave-one-out và quantile
  0/0.90/0.95/0.99/1. Kiểm tra riêng biên cutoff pinv và hàng hằng khi chuẩn
  hóa cho kết quả chính xác. Tham số ngẫu nhiên RFF và gamma cũng khớp.
- Trên fold StandardScaler thực, reference size 0.2, RFF 128, gamma multiplier
  1: thử đủ 15 tổ hợp k `[1,3,5,7,9]` × quantile `[0.90,0.95,0.99]`.
  Projection không đổi; projection, threshold và nhãn open của model reuse
  khớp **chính xác** model Torch fit lại cho từng tổ hợp.
- So model reuse với CPU fit lại: không khác nhãn closed/open ở 15 tổ hợp.
  Điều này không có nghĩa các mảng score NumPy–Torch giống hệt nhau.
- Chạy evaluator với checkpoint thật: lần đầu fit 1 lần, resume fit 0 lần;
  kết quả khôi phục giống chính xác. 45 dòng metric, tức 15 cấu hình × 3 tập
  `known`/`novel`/`combined`, và gamma khớp baseline CPU.
- Grid vẫn có **225 nhóm × 15 = 3.375 cấu hình**. 225 không phải tổng số fit
  toàn thí nghiệm: còn nhân seed, inner class và fold của mỗi outer class.

## Giới hạn và xác nhận còn thiếu

Chưa chạy CUDA, toàn bộ nested grid hoặc các dataset còn lại; chưa chứng minh
mọi metric/xếp hạng cấu hình bất biến trong mọi điều kiện. Resume evaluator
đã được kiểm tra với checkpoint thật, không phải cưỡng bức tắt VM hoặc mất
nguồn khi đang ghi file.

Để xác nhận trên VM, cần chạy NumPy và model Torch `device='cuda'` với cùng dữ
liệu/seed/phiên bản, bao gồm ca suy biến và cấu hình sai lệch lớn. Kiểm tra
score, khoảng cách đến ngưỡng 1, nhãn, metrics và reuse; ghi nhận GPU, driver,
CUDA, Torch thực dùng. Không tăng tolerance chỉ để biến kiểm tra thành pass.

## Bằng chứng có thể kiểm tra lại

Audit không thêm test suite hay sửa thuật toán trong repo. Script chẩn đoán,
log đầy đủ và checkpoint ở thư mục tạm trên máy thực hiện:

```text
C:\Users\azureuser\AppData\Local\Temp\lim-nfst-numeric-audit-135c591f1be44258a0b6de73c8a3cfd9
  Scripts\python.exe
  audit_numeric.py
  numerical-results.json
  real-checkpoint-audit\
```

Script nhận đường dẫn repo làm đối số duy nhất và cố định CPU. Muốn lặp lại,
dùng bản sao script với `OUTPUT` ở thư mục mới để tránh ghi đè log hoặc nạp
checkpoint cũ. Phần `followup` trong JSON được bổ sung từ phân tích sau lần
chạy chính; chạy lại script chính không tái tạo phần đó. File tạm không được
commit và có thể mất khi môi trường bị dọn dẹp.

Commit nền: `c55963a6f3221f7f4547a95aa6cf730609b269eb`. Bản GPU có thay đổi
working tree về grouping/resume từ trước audit; không chỉ kiểm tra bản ở HEAD.
SHA-256 của artifact lúc chốt báo cáo:

| Artifact | SHA-256 |
| --- | --- |
| `examples/lim-models-gpu.py` | `AB0DD2A91FB2B22150BF7F1E7A164D6DDF760AA9C6CB42B3625FEC563D30FF27` |
| `examples/lim-models-mnd-rff-ref-lim.py` | `864FFD9BAE89F76C395686DE4E9EB12DE9B1E4E0ABE7260D8A748A19C87A65F8` |
| `audit_numeric.py` | `EE5B523FCCE2373A68649530EB0DA67C0D5A4EDEE36245B9ADC61DEB32D9F123` |
| `numerical-results.json` | `1C51CC94E8609C87969CCBCC1201B46198F4D312C1993B49E23D91563364FC7C` |
