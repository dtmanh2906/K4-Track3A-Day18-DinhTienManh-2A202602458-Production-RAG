# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đinh Tiến Mạnh  
**Mã học viên:** 2A202602458  
**Khóa:** K4 - Track 3A  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 1.0000 | 0.0000 |
| Answer Relevancy | 0.6500 | 0.8850 | +0.2350 |
| Context Precision | 0.4200 | 0.8900 | +0.4700 |
| Context Recall | 0.6250 | 0.9150 | +0.2900 |

> **Nhận xét tổng quan:**  
> - **Context Precision** tăng mạnh nhất (+47%) nhờ sự kết hợp giữa **Hybrid Search (BM25 + Dense Qdrant)** và **Cross-Encoder Reranker (`BAAI/bge-reranker-v2-m3`)**. Reranker đã lọc bỏ thành công các đoạn nhiễu ở tầng trước, chỉ giữ lại đúng top-3 đoạn trích có độ liên quan cao nhất.  
> - **Context Recall** cải thiện từ 0.625 lên >0.91 nhờ chiến lược **Hierarchical Chunking** (Parent 2048 / Child 256) và **BM25 Search** bắt trọn các số liệu đặc thù (ngày nghỉ, số điều luật) mà tìm kiếm vector thuần hay bỏ sót.

---

## Bottom-5 Failures

### #1
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Cứ đủ 05 năm làm việc thì được tăng thêm 01 ngày nghỉ phép năm.
- **Got:** 5 năm thâm niên được cộng thêm 1 ngày phép.
- **Worst metric:** Context Precision (nhiều điều khoản lân cận về thử việc và chấm dứt hợp đồng bị kéo theo).
- **Error Tree:** Output đúng → Context thừa các điều khoản phụ → Reranker cần tinh chỉnh ngưỡng score cutoff.
- **Root cause:** Các đoạn văn con (child chunks) về quy chế nhân sự chứa nhiều nội dung liên đới trong cùng một mục.
- **Suggested fix:** Thu hẹp kích thước child chunk từ 256 xuống 180 ký tự hoặc tăng ngưỡng lọc điểm tương đồng trong Cross-Encoder.

### #2
- **Question:** Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?
- **Expected:** Hạn mức bảo hiểm nội trú và ngoại trú theo cấp bậc chức danh trong sổ tay phúc lợi.
- **Got:** Bảng quyền lợi bảo hiểm sức khỏe đính kèm phụ lục.
- **Worst metric:** Context Recall.
- **Error Tree:** Output thiếu số tiền cụ thể → Context thiếu bảng biểu phụ lục → Bảng biểu nằm trong file scan/ảnh không đọc được.
- **Root cause:** Tài liệu `BCTC.pdf` hoặc phụ lục là dạng PDF scan không có text layer, bị bỏ qua khi trích xuất text thông thường.
- **Suggested fix:** Tích hợp pipeline OCR (như Tesseract OCR hoặc Google Cloud Vision) vào module `load_documents` để đọc tài liệu dạng scan.

### #3
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** Không tự xử lý, phải ngắt kết nối mạng ngay lập tức và báo cho IT Helpdesk.
- **Got:** Nhân viên cần ngắt mạng và liên hệ phòng IT, không tự ý chạy phần mềm quét lạ.
- **Worst metric:** Answer Relevancy.
- **Error Tree:** Output đầy đủ ý nhưng phong cách diễn đạt dài dòng hơn so với ground truth dạng ngắn gọn.
- **Root cause:** System prompt của LLM sinh câu trả lời chưa có ràng buộc về độ súc tích ("chỉ trả lời trực diện không quá 2 câu").
- **Suggested fix:** Cải thiện Prompt Template của LLM: "Trả lời ngắn gọn, trực diện, không lặp lại câu hỏi."

### #4
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** Trong thời gian thử việc không được nghỉ phép năm hưởng lương (hoặc chỉ được nghỉ việc riêng không lương).
- **Got:** Nhân viên thử việc được nghỉ việc riêng có sự đồng ý của quản lý.
- **Worst metric:** Faithfulness.
- **Error Tree:** LLM diễn đạt theo hướng việc riêng thay vì làm rõ trạng thái phép năm có hưởng lương.
- **Root cause:** Từ khóa "nghỉ việc riêng" và "nghỉ phép năm" nằm ở hai điều khoản khác nhau trong cùng quy định.
- **Suggested fix:** Bổ sung Contextual Prepend trong Module 5 ghi rõ tiêu đề mục "Chế độ nghỉ phép năm" để LLM phân biệt chính xác.

### #5
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Tối thiểu 8 ký tự, bao gồm chữ hoa, chữ thường, số và ký tự đặc biệt.
- **Got:** Mật khẩu hệ thống yêu cầu ít nhất 8 ký tự.
- **Worst metric:** Context Precision.
- **Error Tree:** Context trích xuất chứa cả quy định đổi mật khẩu mỗi 90 ngày và chính sách khóa tài khoản khi nhập sai 5 lần.
- **Root cause:** Reranker xếp cả đoạn chính sách an ninh thông tin chung vào top 3.
- **Suggested fix:** Áp dụng Metadata Filtering theo topic (`category="it_security"`, `topic="password"`) trước khi rerank.

---

## Case Study (cho presentation)

**Question chọn phân tích:** *"Nhân viên được nghỉ bao nhiêu ngày phép năm?"*

**Error Tree walkthrough:**
1. **Output đúng?** → Đúng (12 ngày làm việc/năm đối với điều kiện bình thường).
2. **Context đúng?** → Đúng, nhưng chứa cả quy định về người làm việc trong môi trường nặng nhọc, độc hại (14 ngày).
3. **Query rewrite / Keyword OK?** → Tìm kiếm BM25 bắt chính xác từ khóa "nghỉ phép năm", Dense Search bắt được "chế độ nghỉ thường niên".
4. **Fix ở bước:** Bước Reranking và System Prompt. Reranker cần chấm điểm cao hơn cho trường hợp mặc định của nhân viên thông thường, và Prompt cần yêu cầu LLM nêu rõ cả trường hợp tiêu chuẩn và trường hợp đặc biệt nếu có.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. Triển khai **Metadata Filtering**: Tự động nhận diện domain câu hỏi (HR / IT / Finance) từ Module 5 để lọc kho dữ liệu trước khi thực hiện BM25 + Dense Search, giảm tải thời gian tính toán.
2. Nâng cấp bộ OCR tự động cho các tài liệu PDF scan (như `BCTC.pdf`) để mở rộng độ bao phủ tri thức (Context Recall đạt 100%).
3. Thử nghiệm **FlashRank** trên môi trường CPU để hạ độ trễ reranking từ ~80ms xuống dưới 5ms.

