# 📈 Hệ Thống Kiểm Định Chiến Lược Định Lượng: EMA & OBV (Cổ Phiếu ACB)

Ứng dụng Web App tương tác xây dựng trên nền tảng **Streamlit**, phục vụ việc backtest (kiểm định lịch sử), phân tích tín hiệu kỹ thuật thời gian thực và tối ưu hóa tham số cho chiến lược giao dịch kết hợp giữa **Đường trung bình động lũy thừa (EMA)** và **Chỉ báo khối lượng cân bằng (OBV)** trên dữ liệu 10 năm của cổ phiếu ACB (Ngân hàng TMCP Á Châu).

Dự án được hoàn thiện với đầy đủ các tiêu chuẩn kiểm định định lượng khắt khe: **Hạch toán vị thế (Position Accounting)**, **Triệt tiêu hoàn toàn Look-ahead Bias**, **Ràng buộc số lệnh tối thiểu (MIN_TRADES)** và **Quản trị rủi ro cắt lỗ cứng (Stop Loss 7%)**.

---

## 🌟 Tính Năng Nổi Bật Của Web App

Ứng dụng được thiết kế theo chuẩn giao diện Fintech với **5 Tab chức năng chuyên biệt**:

1. **📊 Tín Hiệu & Biểu Đồ Kỹ Thuật (Live Signals & Interactive Charts):**
   - Đưa ra **khuyến nghị hành động phiên mới nhất** (BUY / HOLD / SELL / STAND ASIDE) dựa trên tổ hợp giữa tín hiệu kỹ thuật và trạng thái vị thế thực tế (đang cầm cổ phiếu hay tiền mặt).
   - Biểu đồ tương tác **Plotly Subplots** phóng to/thu nhỏ, rê chuột xem chi tiết đường giá, đường EMA, điểm khớp lệnh Mua (tam giác xanh), điểm Bán (tam giác đỏ) và đồ thị dòng tiền OBV kèm thanh độ dốc.
   - Bảng theo dõi tín hiệu chi tiết các phiên gần nhất.

2. **🧪 Kiểm Định Hiệu Quả (Train vs Test Backtesting):**
   - Phân tích và so sánh song song 4 chiến lược: **EMA + OBV Kết Hợp**, **EMA Riêng Lẻ**, **OBV Riêng Lẻ** và **Buy & Hold (ACB Benchmark)**.
   - Thẻ KPI đo lường toàn diện: *Total Return (%), Annualized CAGR (%), Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown (%), Win Rate (%), Profit Factor*.
   - Đồ thị đường tăng trưởng vốn (**Equity Curve**) và đồ thị sụt giảm tài sản (**Underwater Drawdown**).
   - Biểu đồ cột so sánh trực tiếp hiệu suất giữa tập **Huấn luyện (Train: 2014–2020)** và tập **Kiểm định độc lập (Test: 2021–2023)** nhằm phát hiện hiện tượng suy giảm hiệu suất (Out-of-sample deterioration).

3. **📋 Quản Lý Vị Thế & Nhật Ký Giao Dịch (Position Accounting & Trade Log):**
   - Trực quan hóa dòng thời gian vị thế danh mục (*0: Tiền mặt, 1: Nắm giữ cổ phiếu*).
   - Xuất bảng nhật ký chi tiết từng lệnh giao dịch đã đóng: *Ngày vào, Giá vào, Ngày ra, Giá ra, Lợi nhuận PnL (%), PnL (VND), Lý do thoát lệnh (Cắt lỗ 7% hay Tín hiệu EMA cắt)*.
   - Nút **tải về file CSV** nhật ký giao dịch chỉ với 1 click.
   - Biểu đồ phân bổ tỷ suất sinh lời của từng lệnh (Histogram).

4. **🎯 Tối Ưu Hóa Tham Số (Hyperopt TPE Optimization):**
   - Tích hợp thuật toán tối ưu hóa thông minh **Tree-structured Parzen Estimator (TPE)** từ thư viện `hyperopt`.
   - Có cơ chế **loại bỏ tham số 'ăn may'**: Ràng buộc `MIN_TRADES >= 5` để triệt tiêu bẫy *"Sharpe ảo / vô cực do ít giao dịch"*.
   - Biểu đồ không gian tham số (Parameter Space vs Sharpe Ratio) trực quan.

5. **📖 Phương Pháp Luận & Khắc Phục Lỗi (Methodology & Fixes):**
   - Trình bày chi tiết lý thuyết, công thức toán học và giải pháp xử lý các lỗi thường gặp trong nghiên cứu định lượng tài chính.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
TAO-APP-CHIEN-LUOC/
│
├── app.py                                   # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt                         # Danh sách thư viện phụ thuộc Python
├── README.md                                # Tài liệu hướng dẫn sử dụng và triển khai
├── ACB.csv                                  # Dữ liệu giá lịch sử 10 năm của ACB (2014 - 2023)
└── ACB_Strategy_EMA_OBV_Nhom3_Adjusted.ipynb # Jupyter Notebook nghiên cứu gốc
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Ứng Dụng Trên Máy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Máy tính đã cài đặt **Python 3.9 trở lên** (khuyên dùng Python 3.10 hoặc 3.11).
- Trình duyệt web hiện đại (Chrome, Edge, Safari, Firefox).

### 2. Các bước khởi chạy
Mở ứng dụng dòng lệnh (**Terminal** trên macOS/Linux hoặc **PowerShell / Command Prompt** trên Windows):

```bash
# Bước 1: Di chuyển vào thư mục dự án
cd "đường-dẫn-đến-thư-mục-dự-án"

# Bước 2: (Khuyên dùng) Tạo và kích hoạt môi trường ảo
# Trên macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Trên Windows:
python -m venv venv
venv\Scripts\activate

# Bước 3: Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# Bước 4: Khởi chạy ứng dụng Streamlit
streamlit run app.py
```

Ứng dụng sẽ tự động mở tab mới trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Đẩy Lên GitHub & Deploy Miễn Phí Trên Streamlit Cloud

### Bước 1: Khởi tạo Git và Đẩy Code Lên GitHub

1. Truy cập vào [GitHub.com](https://github.com) và đăng nhập tài khoản.
2. Tạo một Repository mới (ví dụ đặt tên: `chien-luoc-ema-obv-acb`), chọn chế độ **Public**.
3. Tại Terminal trong thư mục dự án trên máy tính, chạy lần lượt các lệnh sau:

```bash
# Khởi tạo git repository
git init

# Thêm tất cả các file vào git tracking
git add app.py requirements.txt README.md ACB.csv

# Lưu commit đầu tiên
git commit -m "Khởi tạo Web App Kiểm định Chiến lược EMA và OBV cho cổ phiếu ACB"

# Đổi nhánh chính thành main
git branch -M main

# Liên kết với kho chứa trên GitHub (thay URL bằng link repo của bạn)
git remote add origin https://github.com/<tai-khoan-github-cua-ban>/chien-luoc-ema-obv-acb.git

# Đẩy mã nguồn lên GitHub
git push -u origin main
```

---

### Bước 2: Triển Khai (Deploy) Lên Streamlit Cloud

1. Truy cập trang chủ **[share.streamlit.io](https://share.streamlit.io/)** và đăng nhập bằng tài khoản **GitHub**.
2. Nhấn vào nút **"Create app"** (hoặc **"New app"**).
3. Điền các thông tin cấu hình như sau:
   - **Repository:** Chọn `<tai-khoan-github-cua-ban>/chien-luoc-ema-obv-acb`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Bạn có thể tùy chỉnh tên miền miễn phí (ví dụ: `acb-ema-obv-strategy.streamlit.app`).
4. Nhấn nút **"Deploy!"**.
5. Đợi hệ thống đóng gói và triển khai trong 1–2 phút. Khi hoàn tất, web app của bạn sẽ chính thức hoạt động online trên toàn cầu!

---

## 🔬 Phương Pháp Luận & Đóng Góp Khoa Học Của Dự Án

### 1. Quy tắc giao dịch
- **Bộ lọc Xu hướng (Trend):** $EMA(n) = \text{Giá đóng cửa tính theo trọng số hàm mũ chu kỳ } n$.
- **Bộ lọc Dòng tiền (Volume Pressure):** $OBV_t = OBV_{t-1} \pm Volume_t$. Độ dốc dòng tiền: $OBV\_Slope_k = OBV_t - OBV_{t-k}$.
- **Quy tắc Mua (Entry):** $\text{Close}_{t-1} > EMA_{t-1} \text{ VÀ } OBV\_Slope_{t-1} > 0$.
- **Quy tắc Bán (Exit):** $\text{Close}_{t-1} < EMA_{t-1}$ hoặc khi giá chạm ngưỡng **Stop-loss cứng 7%**.

### 2. Các vấn đề học thuật cốt lõi đã được giải quyết
1. **Triệt tiêu Look-ahead Bias:** Tín hiệu được dịch chuyển 1 phiên (`shift(1)`), đảm bảo chỉ dùng dữ liệu đóng nến hôm qua để giao dịch hôm nay.
2. **Hạch toán Vị thế (Position Accounting):** Tài khoản tuân thủ nguyên tắc *Long-only*, chỉ nắm giữ 1 vị thế duy nhất, không nhồi lệnh, không bán khống khi đang giữ tiền mặt.
3. **Loại bỏ Bẫy Sharpe Ảo (Artifact Sharpe Filter):** Ràng buộc số lệnh tối thiểu $N_{trades} \ge 5$ để ngăn thuật toán Hyperopt chọn các bộ tham số 'đứng ngoài thị trường' có phương sai bằng 0 dẫn tới Sharpe vô định ($\text{NaN} / +\infty$).
4. **Chi Phí Giao Dịch Thực Tế:** Đưa vào phí môi giới (0.2%) và trượt giá thị trường (0.1%) cho cả hai chiều giao dịch.

---

## 📊 Kết Quả Thực Nghiệm Tham Khảo (Trên Cổ Phiếu ACB)

| Chiến Lược | Tập Dữ Liệu | Tham Số Tối Ưu | Sharpe Ratio | Tổng Lợi Nhuận (%) | Max Drawdown (%) | Số Lệnh Đã Đóng |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **EMA Riêng lẻ** | **Train (2014-2020)** | EMA=36 | **1.0797** | **+211.97%** | -28.69% | 70 |
| **EMA Riêng lẻ** | **Test (2021-2023)** | EMA=36 | -0.8601 | -37.69% | -59.19% | 41 |
| **OBV Riêng lẻ** | **Train (2014-2020)** | OBV_Slope=19 | **1.1735** | **+277.80%** | -30.46% | 51 |
| **OBV Riêng lẻ** | **Test (2021-2023)** | OBV_Slope=19 | -0.6135 | -38.73% | -62.50% | 38 |
| **⭐ EMA + OBV Kết hợp** | **Train (2014-2020)** | EMA=36, OBV=20 | **1.1514** | **+236.56%** | **-23.87%** | 64 |
| **⭐ EMA + OBV Kết hợp** | **Test (2021-2023)** | EMA=36, OBV=20 | -0.8298 | -36.47% | -60.14% | 37 |

> 📌 **Nhận xét quan trọng:** Chiến lược kết hợp EMA + OBV trên tập Train giúp **giảm thiểu mức sụt giảm tối đa (Max Drawdown giảm từ -28.69% xuống -23.87%)** so với việc chỉ dùng EMA riêng lẻ. Sự suy giảm hiệu suất trong giai đoạn Test (2021–2023) phản ánh trung thực biến động sụt giảm mạnh và chu kỳ downtrend của thị trường chứng khoán Việt Nam trong giai đoạn này, chứng minh tầm quan trọng của việc kiểm định ngoài mẫu (Out-of-sample testing).

---

## 👥 Tác Giả & Bản Quyền

- **Đơn vị phát triển:** Nhóm nghiên cứu & phát triển chiến lược định lượng.
- **Giấy phép:** MIT License (Tự do sử dụng và phát triển vì mục đích học thuật và nghiên cứu).
