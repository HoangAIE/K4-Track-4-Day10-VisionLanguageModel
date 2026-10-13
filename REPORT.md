Họ tên: Ngô Xuân Hoàng
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 0.00 | 0.00 | 0.084 | [0.052, 0.120] | 0.42 | 0.102 | 0.045 | 0.128 |
| E1 | LoRA LLM | 18,874,368 | 8.85 | 24.50 | 0.365 | [0.322, 0.408] | 0.72 | 0.384 | 0.312 | 0.441 |
| E2 | + projector | 127,926,272 | 11.20 | 28.20 | 0.408 | [0.364, 0.451] | 0.68 | 0.425 | 0.358 | 0.486 |
| E3 | + LoRA ViT | 135,102,464 | 13.45 | 33.80 | 0.426 | [0.381, 0.469] | 0.66 | 0.446 | 0.377 | 0.502 |
| E4 | LoRA rank (r=4 / r=64) | 4,718,592 / 75,497,472 | 8.10 / 10.40 | 21.00 / 29.50 | 0.328 / 0.374 | [0.286, 0.371] / [0.331, 0.418] | 0.69 / 0.74 | 0.342 / 0.395 | 0.278 / 0.321 | 0.406 / 0.450 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Phân bổ tham số các khối trong Qwen3-VL-4B (tổng cộng ≈ 4,44B tham số)**:
  1. **Vision Transformer (ViT)**: Gồm 24 khối transformer blocks, chiếm khoảng 415M tham số.
  2. **Vision-Language Projector**: Cầu nối đa tầng giữa ViT và LLM, gồm 2 thành phần chính:
     - `visual.merger`: Giảm số lượng visual tokens (spatial reduction 2x2 merger), chiếm ≈ 27M tham số.
     - `visual.deepstack_merger_list`: Trích xuất đặc trưng đa mức từ các tầng trung gian (tầng 5, 11, 17) của ViT để giữ lại chi tiết không gian độ phân giải cao, chiếm ≈ 82M tham số.
     - Tổng cộng Projector có khoảng 109M tham số (109.051.904 tham số).
  3. **Ngôn ngữ (LLM + Embedding)**: Qwen Language Model gồm 36 tầng transformer layers, chiếm khoảng 4,0B tham số.
- **Khối được cập nhật trong từng thí nghiệm**:
  - **E1 (18.874.368 tham số huấn luyện)**: Chỉ gắn LoRA adapter lên 252 modules của LLM (gồm 36 tầng × 7 modules: self-attention `{q, k, v, o}_proj` và MLP `{gate, up, down}_proj`). ViT và toàn bộ Projector bị đóng băng hoàn toàn (`requires_grad = False`).
  - **E2 (127.926.272 tham số huấn luyện)**: Kế thừa LoRA LLM từ E1, đồng thời mở đầy đủ (full fine-tuning) toàn bộ Projector (`visual.merger` và `visual.deepstack_merger_list`) ở kiểu dữ liệu `float32`. Khối ViT vẫn giữ đóng băng.
  - **E3 (135.102.464 tham số huấn luyện)**: Kế thừa E2 (LoRA LLM + mở full Projector), đồng thời bổ sung thêm LoRA adapter lên 96 modules của ViT (gồm 24 blocks × 4 modules: attention `{qkv, proj}` và MLP `{linear_fc1, linear_fc2}`). Tổng cộng có 348 modules được can thiệp.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **`parse_valid_rate`**: Đạt **0,785 (78,5%)**. Có tới 21,5% số ảnh tập test sinh ra kết quả không thể parse thành danh sách JSON chuẩn theo quy định của `score.py`.
- **Hai lỗi phổ biến nhất của mô hình gốc (E0 zero-shot)**:
  1. *Lỗi cấu trúc hội thoại và text thừa*: Mô hình nền tảng mang bản chất chat instruction, thường tự động thêm các câu dẫn nhập như *"Based on the image, the road damages are located at..."*, hoặc bọc trong markdown code blocks nhưng chèn thêm bình luận cuối chuỗi, khiến hàm trích xuất JSON thất bại hoặc sinh danh sách rỗng.
  2. *Lỗi định dạng toạ độ và sai thang đo*: Mô hình dự đoán toạ độ theo pixel ảnh gốc (vượt quá 640x360), định dạng box không chuẩn (nhầm thứ tự x/y hoặc sinh box có diện tích âm `x2 <= x1`), hoặc bịa ra các box khổng lồ bao trọn toàn bộ bức ảnh do chưa được căn chỉnh grounding miền chuyên biệt.
- **Ví dụ output thô thực tế**:
  `"Certainly! I detected a road issue: ```json [{"label": "pothole", "bbox_2d": [180, 220, 290, 310]}] ``` Let me know if you need more details about road safety."`

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Kết quả F1 khi lệch thang toạ độ**: F1 sụt giảm nghiêm trọng từ **0,365 xuống 0,002** (gần như bằng 0 tuyệt đối).
- **Nguyên nhân toán học**: Ground truth chuẩn hoá toạ độ trên lưới số nguyên [0, 1000] × [0, 1000]. Trong khi đó, ảnh đầu vào dashcam có kích thước vật lý là 640 × 360 pixel.
  - Giả sử một ổ gà có toạ độ pixel thực là `[x1=192, y1=108, x2=320, y2=180]`.
  - Trên thang 0–1000 chuẩn hoá, toạ độ tương ứng phải là `[300, 300, 500, 500]`.
  - Nếu đối chiếu trực tiếp box pixel `[192, 108, 320, 180]` với ground truth thang 1000 `[300, 300, 500, 500]`, vùng giao nhau (intersection) giữa hai hình chữ nhật này là rỗng (IoU = 0.0), hoàn toàn không thể vượt qua ngưỡng `IOU_THR = 0.5`. Hậu quả là toàn bộ dự đoán đúng đều bị coi là False Positive (FP) và toàn bộ nhãn thật thành False Negative (FN), triệt tiêu F1.
- **Kiểm chứng thực tế**: Chạy script `score.py` với tham số `--coord pixel` trên tập dự đoán của E1 (vốn được mô hình sinh ra trên thang 0–1000): kết quả ghi nhận `Precision = 0.003`, `Recall = 0.002`, `F1 = 0.002`.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời**:
  Trong bài toán Vision-Language grounding, prompt hướng dẫn văn bản và 220 tokens đặc trưng thị giác từ ViT đóng vai trò là tiền đề điều kiện (conditioning context). Mục tiêu duy nhất cần tối ưu là phân phối xác suất có điều kiện $P(\text{JSON answer} \mid \text{Image}, \text{Prompt})$. Do đó, tất cả token thuộc prompt và hình ảnh phải được gán nhãn `-100` để `torch.nn.CrossEntropyLoss` bỏ qua trong quá trình tính loss và lan truyền ngược.
- **Điều gì xảy ra nếu tính loss trên toàn bộ chuỗi (không mask)**:
  1. *Mất cân bằng gradient áp đảo*: Mỗi ảnh 640×360 tạo ra 220 visual tokens cộng thêm khoảng 40 token prompt, trong khi chuỗi JSON câu trả lời chỉ chiếm 20–50 tokens (chưa tới 15% tổng độ dài chuỗi). Nếu tính loss toàn bộ, 85% gradient sẽ bị lãng phí vào việc tái tạo lại câu lệnh prompt cố định và phục dựng token ảnh.
  2. *Suy giảm năng lực grounding*: Mô hình bị phân tâm khỏi nhiệm vụ cốt lõi là sinh toạ độ hộp giới hạn chính xác, dẫn đến hiện tượng hallucination (sinh toạ độ ảo), loss giảm nhưng F1 trên tập kiểm thử sụt giảm nghiêm trọng.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit**:
  Nếu để BitsAndBytes lượng tử hoá toàn bộ mô hình, các trọng số của Projector (`visual.merger`) sẽ bị đóng băng thành các tensor lượng tử hoá NF4 tĩnh (`bnb.nn.Linear4bit`). Trọng số 4-bit không hỗ trợ tích luỹ gradient trực tiếp độ chính xác cao khi cập nhật tham số thông thường (full gradient descent).
- **Cách xử lý**:
  Thiết lập `llm_int8_skip_modules=["visual", "lm_head"]` trong `BitsAndBytesConfig`. Cấu hình này chỉ định BitsAndBytes bỏ qua toàn bộ khối thị giác (bao gồm ViT và Projector), giữ nguyên chúng ở độ chính xác ban đầu `float16`. Nhờ đó, ta có thể dễ dàng giải phóng (`requires_grad = True`) các tầng của projector để huấn luyện đầy đủ ở E2 và E3.
- **Vì sao projector phải ở float32**:
  Projector đóng vai trò cầu nối ánh xạ giữa hai không gian biểu diễn rất khác biệt (ViT tokens và không gian embedding ngôn ngữ). Trong quá trình gradient backpropagation với learning rate $2 \times 10^{-4}$, biên độ gradient chạy qua merger và deepstack layers rất lớn. Nếu giữ ở `float16`, dải biểu diễn hạn chế của fp16 dễ gặp hiện tượng tràn số (overflow sinh ra `NaN`/`Inf`) hoặc triệt tiêu số (underflow). Ép kiểu projector về `torch.float32` đảm bảo độ ổn định số học tối đa.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên sẽ cho F1 cao giả (Rò rỉ dữ liệu - Data Leakage)**:
  Tập dữ liệu gồm 2.009 bức ảnh được cắt liên tục từ các video dashcam hành trình (với tần suất khung hình cao). Hai frame liền kề trong cùng một video chỉ cách nhau vài phần trăm giây nên có bối cảnh mặt đường, góc quay, vị trí nắp cống hay ổ gà gần như giống nhau hoàn toàn. Nếu chia ngẫu nhiên, các frame trùng lặp này sẽ xuất hiện đồng thời ở cả tập train và tập test. Khi đó, mô hình chỉ việc "ghi nhớ vẹt" hình ảnh đã thấy ở tập train là có thể dự đoán đúng ở test, tạo ra F1 cao giả tạo (> 0,75). Chia theo phiên quay (`session_key`) gom toàn bộ frame của cùng một buổi quay vào duy nhất một split, buộc mô hình phải học cách tổng quát hoá trên những đoạn đường hoàn toàn mới.
- **Phân bố số ảnh và số box từng lớp theo từng split**:
  - **Train (1.422 ảnh)**: 1.074 pothole · 1.750 crack · 691 manhole (Tổng cộng: 3.515 box).
  - **Val (276 ảnh)**: 66 pothole · 470 crack · 103 manhole (Tổng cộng: 639 box).
  - **Test (311 ảnh)**: 121 pothole · 299 crack · 163 manhole (Tổng cộng: 583 box).
  *Nhận xét*: Do chia theo phiên quay thực tế, tỷ lệ phân bố giữa các split không hoàn toàn đồng đều (ví dụ tập val có rất ít pothole - chỉ 66 box). Đây là thách thức thực tế đòi hỏi phải đánh giá chi tiết theo F1 từng lớp và khoảng tin cậy CI95.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh số liệu thực nghiệm giữa 3 setup**:
  - **E1 (LoRA LLM)**: F1 = 0,365 [CI95: 0,322 – 0,408], VRAM đỉnh = 8,85 GB, Thời gian = 24,5 phút.
  - **E2 (+ mở Projector)**: F1 = 0,408 [CI95: 0,364 – 0,451], VRAM đỉnh = 11,20 GB, Thời gian = 28,2 phút.
  - **E3 (+ LoRA ViT)**: F1 = 0,426 [CI95: 0,381 – 0,469], VRAM đỉnh = 13,45 GB, Thời gian = 33,8 phút.
- **Phân tích đánh giá**:
  - *Mở Projector (E2)* là bước cải tiến **RẤT ĐÁNG GIÁ**: F1 tăng vọt từ 0,365 lên 0,408 (+0,043 F1, tương đương +4,3 điểm phần trăm). VRAM chỉ tăng thêm 2,35 GB (vẫn nằm trong ngưỡng an toàn 16 GB của T4) và chỉ tốn thêm 3,7 phút huấn luyện. Việc căn chỉnh lại không gian biểu diễn giữa ảnh và chữ giúp mô hình hiểu sâu sắc hơn toạ độ bounding box.
  - *Mở thêm LoRA trên ViT (E3)* mang lại hiệu quả **CẦN CÂN NHẮC KỸ**: Mặc dù F1 tăng nhẹ thêm 0,018 (lên 0,426), VRAM đỉnh tăng vọt lên 13,45 GB (tiệm cận giới hạn OOM của T4) và thời gian tăng thêm 5,6 phút. Đặc biệt, khoảng tin cậy CI95 của E2 [0,364 – 0,451] và E3 [0,381 – 0,469] có độ chồng lấn (overlap) rất lớn, cho thấy mức chênh lệch 0,018 chưa đủ sức thuyết phục về mặt ý nghĩa thống kê trên tập test 311 ảnh. Do đó, xét về bài toán tối ưu chi phí tài nguyên / hiệu năng thực tế, **E2 là lựa chọn tối ưu nhất**.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy bootstrap CI95**:
  Trên tập test 311 ảnh với 583 box, khoảng tin cậy 95% có độ rộng dao động khoảng **±0,043 đến ±0,045** (tổng bề rộng dải CI đạt xấp xỉ **0,086 – 0,088**). Cụ thể: E1 là [0,322 – 0,408] (dải rộng 0,086), E2 là [0,364 – 0,451] (dải rộng 0,087).
- **Giới hạn kết luận so sánh**:
  - Khoảng cách giữa E0 (0,084, CI: [0,052 – 0,120]) và E1 (0,365, CI: [0,322 – 0,408]) là hoàn toàn tách biệt, cho phép khẳng định tuyệt đối rằng LoRA finetuning vượt trội hơn hẳn Zero-shot.
  - Tuy nhiên, giữa E2 (0,408) và E3 (0,426), do dải CI95 rộng gần 0,09 và hai khoảng tin cậy chồng lấn sâu lên nhau, ta **không thể kết luận chắc chắn 100% rằng E3 sẽ luôn tốt hơn E2** trong mọi tình huống thực tế; chênh lệch 0,018 có thể bắt nguồn từ phương sai ngẫu nhiên trong việc phân mẫu của tập test 311 ảnh. Cần một tập test lớn hơn (≥ 1.000 ảnh) để khẳng định tính vượt trội của E3.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp có hiệu năng kém nhất**: Lớp **`crack` (vết nứt)** có F1 thấp nhất xuyên suốt các thí nghiệm (E1: 0,312 · E2: 0,358 · E3: 0,377), thấp hơn đáng kể so với `pothole` (0,425 ở E2) và `manhole` (0,486 ở E2).
- **Nguyên nhân chi tiết**:
  1. *Đặc điểm kích thước và hình học (Aspect Ratio cực đoan)*: Vết nứt trên mặt đường thường dài, mảnh và ngoằn ngoèo. Khi mô hình dự đoán hộp bao chữ nhật (`bbox_2d`), diện tích box thường rất mỏng. Với ngưỡng khắt khe `IoU ≥ 0.5`, chỉ cần dự đoán lệch một vài pixel theo chiều rộng là diện tích giao nhau tụt giảm nghiêm trọng, khiến IoU rớt xuống dưới 0,5.
  2. *Độ phức tạp ngữ nghĩa thị giác*: Vết nứt có độ tương phản thấp, rất dễ bị nhầm lẫn với vệt bóng râm của cây cối, vệt bánh xe phanh gấp hoặc đường nối nhựa đường. Ngược lại, nắp cống (`manhole`) có kết cấu hình học tròn/vuông chuẩn xác, vật liệu kim loại nổi bật nên mô hình đạt F1 cao nhất (~0,49).
- **Danh sách 6 ảnh minh hoạ thực nghiệm**:
  1. `vlcsnap-2025-02-18-23h08m14s528.jpg`: Dự đoán đúng nắp cống tròn (True Positive, IoU = 0.82).
  2. `vlcsnap-2025-02-18-18h12m44s102.jpg`: Dự đoán đúng ổ gà sâu trung tâm (True Positive, IoU = 0.74).
  3. `vlcsnap-2025-02-19-14h05m22s310.jpg`: Bỏ sót vết nứt mờ mép đường (False Negative lớp crack).
  4. `vlcsnap-2025-02-19-14h20m11s640.jpg`: Nhầm lẫn vệt bóng râm thành vết nứt (False Positive lớp crack).
  5. `vlcsnap-2025-02-18-17h45m09s880.jpg`: Phát hiện ổ gà nhưng box bị lệch nhẹ khiến IoU = 0.44 (< 0.5) dẫn đến bị tính thành 1 FP và 1 FN.
  6. `vlcsnap-2025-03-16-15h16m59s824.jpg`: Dự đoán hoàn hảo cả ổ gà và nắp cống trên cùng một khung hình.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được lựa chọn**: Khảo sát độ nhạy của siêu tham số LoRA Rank (`r`), so sánh **`r = 4`** (alpha = 8) và **`r = 64`** (alpha = 128) so với baseline E1 là **`r = 16`** (alpha = 32). Giữ nguyên toàn bộ các siêu tham số khác (learning rate 2e-4, batch size 2, grad accum 4, 300 steps).
- **Kết quả thu được**:
  - **`r = 4` (`e4_rank_4.json`)**: 4,7M tham số, VRAM đỉnh = 8,10 GB, F1 = **0,328** [CI95: 0,286 – 0,371].
  - **`r = 16` (E1 baseline)**: 18,9M tham số, VRAM đỉnh = 8,85 GB, F1 = **0,365** [CI95: 0,322 – 0,408].
  - **`r = 64` (`e4_rank_64.json`)**: 75,5M tham số, VRAM đỉnh = 10,40 GB, F1 = **0,374** [CI95: 0,331 – 0,418].
- **Đánh giá độ tin cậy**:
  - Việc giảm rank xuống $r = 4$ làm giảm đáng kể năng lực học của mô hình (F1 giảm 0,037), chứng minh rank thấp không đủ dung lượng để thích ứng với tác vụ visual grounding phức tạp.
  - Tăng rank lên $r = 64$ giúp F1 tăng nhẹ thêm 0,009 (lên 0,374), nhưng số tham số huấn luyện tăng vọt gấp 4 lần. Khoảng tin cậy CI95 của $r = 16$ [0,322 – 0,408] và $r = 64$ [0,331 – 0,418] bao phủ nhau gần như hoàn toàn. Do đó, kết luận tăng rank từ 16 lên 64 mang lại cải thiện vượt bậc là **chưa đáng tin cậy**; $r = 16$ là điểm dung hoà lý tưởng nhất.

## Nhìn lại

Bài lab đã đem lại cái nhìn sâu sắc và thực tế về bài toán Vision-Language Fine-tuning:
1. Hiểu rõ cơ chế phối hợp giữa ViT, Projector (merger & deepstack), và LLM; nhận ra tầm quan trọng của việc huấn luyện Projector ở độ chính xác float32 để tránh tràn số.
2. Thấy được tác hại của hiện tượng rò rỉ dữ liệu video và tầm quan trọng của việc chia split theo session quay.
3. Nhận ra rằng với tập dữ liệu kiểm thử kích thước nhỏ (311 ảnh), việc luôn đọc số liệu kèm khoảng tin cậy bootstrap CI95 là bắt buộc để tránh đưa ra những nhận định cảm tính về mô hình.
