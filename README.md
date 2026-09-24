# 🌱 LoGVN — Prototype tối ưu tuyến thu gom chất thải rắn sinh hoạt
## Khu vực nghiên cứu: Tuyến Lê Văn Việt và lân cận, TP. Thủ Đức, TP.HCM

## 0. Cập nhật giao diện & tính năng (bản LoGVN)

Giao diện `app.py` đã được thiết kế lại thành **2 khu vực** ở phần thân trang
(không còn dùng sidebar), theme màu xanh lá (`.streamlit/config.toml`):

- **Khu I (trái, có thể thu gọn bằng toggle "◀ Thu gọn khu I")**: (1) Dữ liệu
  học máy — upload Excel lịch sử + nút "🎯 Huấn luyện mô hình"; (2) Dự báo —
  chọn ngày dự báo (time window theo lịch thực tế, lấy từ chính horizon vừa
  huấn luyện) + nút "▶️ Chạy dự báo"; nguồn điểm thu gom (demo/upload).
- **Khu II (phải)**: (3) Routing (OSRM); (4) Ràng buộc tải trọng xe — **không
  còn giới hạn thủ công số xe**, thay vào đó chỉnh "Tải trọng tối đa cho phép
  / xe (%)" (mặc định 115%), số xe cần thiết được tự tính bằng
  `auto_fleet_size()` = `ceil(tổng khối lượng / (capacity × %)) + 1 xe dự
  phòng`, OR-Tools tự thêm xe nếu vẫn chưa khả thi; (5) Thuật toán OR-Tools,
  có thêm lựa chọn baseline (Nearest Neighbor hoặc Clarke-Wright Savings);
  (6) Hệ số tiêu hao/chi phí/phát thải, có thêm **đơn giá nhiên liệu (VNĐ/lít)**
  để quy đổi ra **chi phí**, và nút xuất báo cáo Excel sau khi chạy tối ưu.
- **Dashboard so sánh** (thân trang, sau khi chạy tối ưu): quãng đường, chi
  phí nhiên liệu, số xe, CO2 — Baseline vs Optimized, kèm % tiết kiệm.
- **Bảng phân bổ xe theo điểm**: danh sách điểm từng xe phụ trách, tải trọng
  (%) tính trên ngưỡng 115% (không phải trên capacity định mức — có chú
  thích rõ trên giao diện).
- **🚧 Mô phỏng sự cố giao thông**: chọn 1 điểm bị sự cố + hệ số nhân thời
  gian di chuyển qua điểm đó, tính lại toàn bộ tuyến ngay lập tức và so sánh
  trước/sau. **Đây là mô phỏng thủ công**, KHÔNG phải traffic feed thời gian
  thực từ nhà cung cấp bản đồ (OSRM demo server công khai không có dữ liệu
  này miễn phí).
- **Thuật toán Dynamic (chặng phụ)**: trong mục GPS tự động, tài xế có thể
  báo "🚧 sự cố" tại 1 điểm đang pending (vật cản, rác cồng kềnh, không quay
  đầu được...) — điểm bị chuyển sang trạng thái `deferred`, loại khỏi tuyến
  hiện tại, tuyến còn lại được tính lại ngay. Khi xe về DEPOT mà vẫn còn điểm
  `deferred` và xe **chưa đầy tải** (so với capacity), hệ thống tự động tạo
  **1 chặng phụ** (DEPOT → các điểm bị hoãn → DEPOT). Logic này nằm trong
  `dynamic_routing.py` (`mark_deferred`, `requeue_deferred`,
  `deferred_node_ids`).

Các module xử lý (`baseline.py`, `optimizer.py`, `clustering.py`,
`waste_streams.py`, `forecasting.py`, `routing.py`) không đổi API, chỉ
`dynamic_routing.py` được bổ sung 3 hàm nói trên.

## 1. Kiến trúc code

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit, điều phối toàn bộ luồng xử lý |
| `routing.py` | Toàn bộ chức năng OSM/OSRM (distance/duration matrix, road geometry), cache, xử lý lỗi |
| `baseline.py` | Baseline heuristic Nearest Neighbor / Greedy và Clarke-Wright Savings (có capacity, service time, depot) |
| `optimizer.py` | OR-Tools CVRP/VRPTW + Guided Local Search |
| `clustering.py` | Mô hình đề xuất "Cluster-first, Route-second": K-means++ phân cụm (tự động số cụm k, cân bằng tải trọng) + CVRP theo từng cụm |
| `waste_streams.py` | Tối ưu tuyến riêng cho từng luồng rác (tái chế / thực phẩm / còn lại), tái sử dụng ma trận OSRM đã cache |
| `forecasting.py` | Dự báo nhu cầu rác (XGBoost / Facebook Prophet) từ lịch sử 180 ngày, xuất demand cho routing |
| `dynamic_routing.py` | Theo dõi GPS xe thời gian thực: auto-completion điểm thu gom, auto depot detection, tự động re-optimize khi xe về depot; báo sự cố/vật cản (`mark_deferred`) và tự động tạo chặng phụ (`requeue_deferred`) |
| `data_generator.py` | Sinh dữ liệu demo trong phạm vi Lê Văn Việt; chuẩn hoá dữ liệu upload |
| `requirements.txt` | Thư viện cần cài |

## 2. Cài đặt

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Lưu ý về Prophet:** `prophet` (Facebook/Meta) cần trình biên dịch C++ và
> có thể mất vài phút để cài/biên dịch `cmdstanpy` lần đầu, tuỳ hệ điều hành.
> Nếu chỉ cần demo nhanh, có thể dùng phương án XGBoost (không cần biên dịch
> gì thêm) và bỏ qua Prophet trong lần cài đầu tiên.

## 3. Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại địa chỉ Streamlit hiển thị (mặc định `http://localhost:8501`).

> **Lưu ý về OSRM:** Ứng dụng mặc định gọi OSRM demo server công khai
> (`https://router.project-osrm.org`). Server này có giới hạn tốc độ và chỉ
> phù hợp cho mục đích thử nghiệm/nghiên cứu. Nếu cần dùng ổn định/số lượng
> lớn, nên tự dựng OSRM server (Docker + dữ liệu OSM khu vực TP.HCM) và đổi
> "OSRM base URL" trong sidebar sang địa chỉ server riêng.

## 4. Chuẩn bị file CSV/XLSX (nếu không dùng dữ liệu demo)

Cột bắt buộc:

| Cột | Ý nghĩa |
|---|---|
| `node_id` | Mã điểm (duy nhất) |
| `latitude` | Vĩ độ |
| `longitude` | Kinh độ |
| `waste_kg` | Khối lượng rác phát sinh tại điểm (kg) |

Cột tuỳ chọn:

| Cột | Ý nghĩa | Mặc định nếu thiếu |
|---|---|---|
| `service_time` | Thời gian phục vụ tại điểm (phút) | 5 |
| `time_window_start` | Bắt đầu khung giờ được phép thu gom (phút, tính từ 00:00) | 0 |
| `time_window_end` | Kết thúc khung giờ (phút) | 1440 |
| `is_depot` | `True` cho đúng 1 hàng là depot | hàng đầu tiên |

Tất cả toạ độ nên nằm trong/quanh khu vực Lê Văn Việt, TP. Thủ Đức để đúng
phạm vi nghiên cứu đã xác định.

## 5. Luồng xử lý (pipeline)

```
Data (demo hoặc upload)
   -> OSM/OSRM (Table Service)
   -> Distance matrix (m) + Time matrix (s)   [ma trận CHÍNH cho toàn bộ mô hình]
   -> Baseline (Nearest Neighbor / Greedy, cùng ma trận OSRM)
   -> OR-Tools (CVRP / VRPTW)
   -> Guided Local Search (metaheuristic)
   -> Optimized Routes
   -> OSRM Route Service -> Road geometry thực tế cho từng tuyến
   -> Folium Map (baseline vs optimized)
   -> Bảng KPI (distance, time, số xe, capacity utilization, fuel, CO2)
   -> So sánh % giảm (distance / time / CO2)
```

## 6. Các nguyên tắc đã áp dụng theo yêu cầu

- **Không dùng Haversine làm khoảng cách chính** – chỉ dùng khi người dùng
  chủ động bật "cho phép fallback Haversine", và giao diện sẽ hiển thị rõ
  cảnh báo "Fallback mode – không sử dụng mạng lưới đường thực tế".
- **Không dùng `distance / average_speed` làm travel time chính** – thời
  gian di chuyển luôn lấy từ ma trận `duration` của OSRM Table Service.
- **Baseline và Optimized dùng chung một ma trận OSRM** để đảm bảo so sánh
  công bằng.
- **CO2 là giá trị ước tính** ("Estimated CO2 emissions"), tính từ
  `distance × fuel_rate × emission_factor`; các hệ số này có thể chỉnh trong
  sidebar.
- **Không âm thầm chuyển sang Haversine khi OSRM lỗi** – mặc định báo lỗi rõ
  ràng: *"Không thể lấy dữ liệu mạng lưới đường từ OSRM. Vui lòng thử lại."*
- **Cache OSRM** bằng `st.cache_data` cho distance/duration matrix và cho
  road geometry, tránh gọi lại OSRM không cần thiết mỗi lần Streamlit rerun.
- **Dữ liệu demo** chỉ được sinh quanh các waypoint dọc tuyến Lê Văn Việt
  (20–30 điểm + 1 depot, mặc định 2–3 xe), bán kính lệch tối đa ~200m để mô
  phỏng các hẻm/đường nhánh kết nối trực tiếp với trục chính — không rải ra
  toàn TP.HCM/TP. Thủ Đức.
- Nhãn khu vực hiển thị trên giao diện: **"Khu vực thử nghiệm: Tuyến Lê Văn
  Việt – TP. Thủ Đức, TP.HCM"**.

## 7. Giới hạn của prototype (đúng với mục tiêu đề tài sinh viên)

- Chưa xét traffic thời gian thực.
- Toạ độ waypoint dọc Lê Văn Việt trong `data_generator.py` là toạ độ tham
  khảo/xấp xỉ để mô phỏng khu vực nghiên cứu, không phải kết quả khảo sát
  thực địa chính xác từng mét — nếu dùng cho báo cáo chính thức, nên thay
  bằng toạ độ khảo sát thực tế (qua file CSV/XLSX upload).
- OR-Tools + Guided Local Search là thuật toán tối ưu duy nhất được sử dụng,
  không thay thế bằng thuật toán AI khác.
- `optimizer.py` đã hỗ trợ sẵn đội xe không đồng nhất (`vehicle_capacities_kg`)
  và ràng buộc "xe nhỏ cho hẻm sâu" (`small_vehicle_only_nodes`,
  `small_vehicle_flags`) ở tầng hàm số; `data_generator.py` cũng đã sinh sẵn
  cột `requires_small_vehicle`. Tuy nhiên `app.py` hiện **chưa có control nào
  ở sidebar** để bật ràng buộc này khi gọi tối ưu chính — cột
  `requires_small_vehicle` mới chỉ được hiển thị tham khảo trong bảng chi
  tiết, chưa được truyền vào `OptimizeConfig`. Đây là hướng mở rộng UI ở
  phiên bản sau, không phải lỗi của các module xử lý.

## 8. Dynamic Routing (GPS tự động) — tính năng mới

Ở cuối trang có section **"🛰️ Dynamic Routing – GPS tự động"** theo dõi 1 xe
theo thời gian thực và tự động re-optimize, KHÔNG cần tài xế bấm nút xác nhận:

- **Auto completion**: xe vào bán kính 30–50m quanh 1 điểm thu gom và đứng
  yên trong vùng đó tối thiểu 10 giây (mặc định, có thể chỉnh) → điểm tự
  chuyển `pending → completed`.
- **Auto depot detection**: xe vào bán kính 30–50m quanh DEPOT và đứng đủ
  thời gian → hệ thống tự xác nhận "xe đã về DEPOT", không cần bấm nút.
- **Auto re-optimization**: ngay khi xác nhận về DEPOT, hệ thống lấy toàn bộ
  điểm còn `pending`, đặt DEPOT làm điểm xuất phát, tự động chạy lại
  OR-Tools + Guided Local Search và hiển thị tuyến mới trên bản đồ.
- **Không suy luận "xe đầy" từ GPS**: tín hiệu duy nhất để tái tối ưu là
  "xe đã về DEPOT" (dwell tại depot đủ lâu). Không có cảm biến tải trọng
  trong phạm vi prototype này.

Có 2 nguồn GPS:

1. **Mô phỏng GPS (demo/test)** — xe di chuyển ảo dọc theo road geometry của
   tuyến hiện tại với tốc độ và hệ số tăng tốc do người dùng chọn. Đây là
   cách khuyến nghị để demo/kiểm chứng logic auto-completion/auto
   re-optimize mà không cần xe thật.
2. **GPS thực từ điện thoại (thử nghiệm)** — dùng package
   `streamlit-geolocation` để đọc vị trí trình duyệt. **Giới hạn quan trọng**:
   trình duyệt yêu cầu quyền định vị và, tuỳ thiết bị/trình duyệt, có thể cần
   tương tác lại để cấp phép hoặc lấy vị trí mới — đây là giới hạn bảo mật
   của trình duyệt web (không có API JavaScript nào theo dõi GPS nền hoàn
   toàn im lặng trên mọi trình duyệt/thiết bị di động). Nếu cần độ tin cậy
   cao hơn cho vận hành thực tế, nên dùng app di động gốc (native) gửi GPS
   định kỳ qua API riêng.

Module logic (`dynamic_routing.py`) hoàn toàn tách biệt khỏi Streamlit nên có
thể unit-test độc lập (xem cách test trong quá trình phát triển) và tái sử
dụng cho backend khác nếu cần mở rộng sau này.

## 9. Tối ưu đa luồng theo đề án phân loại rác tại nguồn — tính năng mới

Module mới `waste_streams.py` mô hình hoá đúng lộ trình phân loại rác thực tế
(Luật Bảo vệ môi trường 2020, Quyết định 63/2024/QĐ-UBND TP.HCM):

- Đa số điểm (hộ gia đình): phân **2 nhóm** — Tái chế / Còn lại.
- Nhóm "chủ nguồn thải phát sinh nhiều rác thực phẩm" (chợ, nhà hàng, khách
  sạn, TTTM có dịch vụ ăn uống — đánh dấu bằng cột `is_major_food_generator`
  trong dữ liệu demo/upload): phân **3 nhóm** — thêm luồng Thực phẩm riêng.

Mỗi luồng (`recyclable`, `food`, `other`) được coi là **một bài toán CVRP độc
lập**, có thể cấu hình số xe/tải trọng xe riêng (VD xe rác thực phẩm thường
nhỏ hơn, kín mùi), nhưng vẫn dùng lại đúng OR-Tools + Guided Local Search và
**tái sử dụng ma trận OSRM đã cache** từ lần chạy tối ưu chính (không gọi lại
OSRM cho từng luồng) để tiết kiệm request và đảm bảo so sánh công bằng.

Section **"♻️ Tối ưu tuyến theo từng luồng rác"** trong `app.py` hiển thị:
- KPI Baseline vs Optimized riêng cho từng luồng.
- Bảng so sánh **"Trước phân loại (1 luồng gộp)" vs "Sau phân loại (tổng các
  luồng riêng)"** — minh hoạ đánh đổi thực tế: tách luồng thường làm tăng
  tổng quãng đường/số xe, nhưng đổi lại tách bạch được dòng rác tái chế (miễn
  phí thu gom theo biểu giá TP.HCM hiện hành) và rác thực phẩm (giảm khối
  lượng chôn lấp).

**Giới hạn**: đây là mô phỏng tại 1 thời điểm (không mô hình hoá lịch thu gom
nhiều ngày/tần suất khác nhau giữa các luồng — ví dụ rác thực phẩm thường cần
thu hằng ngày còn tái chế 2–3 lần/tuần). Nếu cần mô hình hoá lịch thu gom định
kỳ, nên mở rộng sang bài toán Periodic VRP (PVRP) ở phiên bản sau.

## 10. So sánh Clustering (K-means++) vs không phân cụm — tính năng mới

Module mới `clustering.py` triển khai mô hình "Cluster-first, Route-second":

1. **Giai đoạn phân cụm**: K-means với khởi tạo **k-means++**, số cụm k được
   xác định **tự động** theo công thức `k = ⌈ΣTᵢ / (Q × safety_margin)⌉`
   (safety_margin mặc định 0,9 để chừa dư địa cho bước cân bằng tải trọng).
   Sau phân cụm không gian, một bước **cân bằng tải trọng** (capacity
   balancing) sẽ chuyển các điểm ở cụm quá tải sang cụm liền kề còn dư sức
   chứa, ưu tiên cụm gần nhất, cho đến khi mọi cụm ≤ capacity xe (hoặc hết
   khả năng cân bằng nếu dữ liệu quá khít).
2. **Giai đoạn định tuyến**: mỗi cụm được giải như MỘT bài toán CVRP độc lập
   bằng OR-Tools (PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH), tái sử dụng đúng
   `optimizer.py`, rồi gộp kết quả các cụm lại thành lời giải tổng thể.

Section **"🧩 So sánh: Đường đi ngắn nhất · CVRP trực tiếp · Mô hình đề
xuất"** trong `app.py` sinh một **bộ dữ liệu mô phỏng riêng (mặc định 100
điểm)** dọc Lê Văn Việt, độc lập với dữ liệu ở phần tối ưu chính, rồi chạy và
so sánh 3 phương pháp trên CÙNG một ma trận OSRM:

1. Đường đi ngắn nhất (Nearest Neighbor), không phân cụm.
2. CVRP trực tiếp (OR-Tools + GLS) trên toàn bộ điểm, không phân cụm.
3. Mô hình đề xuất (K-means++ phân cụm + CVRP theo từng cụm).

Bảng kết quả gồm số cụm, số xe, quãng đường, % giảm so với phương án 1, và
**runtime thuật toán** — ngân sách thời gian được **chia đều cho từng cụm**
(`split_time_budget`) để so sánh runtime với phương pháp không phân cụm là
công bằng.

**Phát hiện thực nghiệm đáng lưu ý** (đã kiểm chứng khi phát triển, dùng ma
trận Haversine thay OSRM để test logic offline): với bộ 100 điểm demo, CVRP
trực tiếp cho kết quả quãng đường TỐT HƠN mô hình phân cụm, và mô hình phân
cụm có runtime CAO HƠN CVRP trực tiếp khi giải tuần tự (mỗi cụm cần tối thiểu
~2 giây để OR-Tools khởi tạo, nhân với số cụm k). Đây không phải lỗi mà là
đặc điểm cố hữu của cách tiếp cận "chia để trị" khi giải TUẦN TỰ — trong triển
khai thực tế, các cụm là bài toán ĐỘC LẬP nên có thể giải SONG SONG
(parallelization) để runtime thực tế giảm gần bằng runtime của cụm chậm nhất
thay vì tổng cộng dồn lại; đây là điểm nên đưa vào phần thảo luận/hạn chế của
đề tài thay vì mặc định cho rằng phân cụm luôn nhanh hơn/tốt hơn.

## 11. Dự báo từ Excel 180 ngày → định tuyến

Giao diện hiện có pipeline theo thứ tự:

1. **Dữ liệu điểm thu gom**: CSV/XLSX chứa `node_id`, `latitude`, `longitude`, `waste_kg` và các cột tuỳ chọn.
2. **Dự báo nhu cầu**: upload file lịch sử (khuyến nghị 180 ngày), chọn XGBoost hoặc Prophet và chọn horizon 1–14 ngày.
3. **Xuất Excel dự báo**: workbook gồm lịch sử chuẩn hoá, dự báo theo điểm/ngày và kiểm định mô hình.
4. **Định tuyến**: demand của ngày dự báo được truyền trực tiếp sang baseline Nearest Neighbor và OR-Tools + GLS; không cần upload lại file dự báo.
5. **Phân bổ xe & báo cáo**: giao diện và Excel có bảng Daily Collection Operations theo cấu trúc Table 4, cùng bảng Vehicle Allocation.

### Định dạng file lịch sử

Tối thiểu: `date`, `node_id`, `waste_kg`. Nếu có dữ liệu phân loại, dùng thêm `waste_organic_kg`, `waste_recyclable_kg`, `waste_other_kg`. `num_households` là cột tuỳ chọn. Nếu chỉ có `waste_kg`, hệ thống tạm phân bổ thành phần 55% hữu cơ / 20% tái chế / 25% còn lại để chạy mô hình đa luồng; đây là giả định của prototype, không phải dữ liệu đo thực tế.
