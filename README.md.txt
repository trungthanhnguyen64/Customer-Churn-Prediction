# DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ
Nhóm 8 - Nhập môn trí tuệ nhân tạo - 20221 - Đại học Bách Khoa Hà Nội
## Thành viên nhóm
* Đàm Việt Anh - 20204627 - anh.dv204627@sis.hust.edu.vn
* Vũ Việt Anh - 20200053 - anh.vv200053@sis.hust.edu.vn
* Vũ Đức Quỳnh - 20204684 - quynh.vd204684@sis.hust.edu.vn
* Nguyễn Đăng Khoa - 20204572 - khoa.nd@sis.hust.edu.vn
* Trịnh Quang Quân - 20200511 - quan.tq@sis.hust.edu.vn
### Lời cảm ơn
  Đây là tệp mã nguồn đồ án môn học Nhập môn trí tuệ nhân tạo, lớp Nhập môn trí tuệ nhân tạo do PGS. TS Thân Quang Khoát phụ trách. Lời đầu tiên, chúng em xin gửi lời cảm ơn đến giảng viên bộ môn là thầy Thân Quang Khoát, đã nhiệt tình giảng dạy và góp ý để chúng em hoàn thành bài tập lớn môn học. Những bài giảng trên nền tảng youtube của thầy là nguồn gợi ý thiết thực cho việc tìm các giải pháp cải thiện cho bài toán. Chúng em xin phép được gắn link youtube bài giảng của thầy tại đây: https://www.youtube.com/@thanquangkhoat4070. Cuối cùng, nhóm chúng em xin kính chúc thầy có thật nhiều sức khỏe, hành phúc và luôn thành công trong sự nghiệp giảng dạy của mình.
## Hướng dẫn 
* Đường link đến tập dữ liệu: https://www.kaggle.com/code/vsridevi/capstone-project-churn-prediction/data?scriptVersionId=110749928
* Mô tả:
  Một nhà cung cấp dịch vụ DTH (Direct to Home) trên thị trường đang phải đối mặt với những thách thức để giữ chân khách hàng hiện tại do có sự cạnh tranh gay gắt trong tình hình hiện tại. Trong công ty này, một tài khoản có thể có nhiều khách hàng được gắn thẻ, vì thế khi mất một tài khoản, công ty có thể mất nhiều hơn một khách hàng. Đồng thời, chi phí để quáng cáo sản phẩm tới những khách hàng mới thường đắt đỏ hơn với việc giữ chân khách hàng cũ. Do đó, tài khoản rời đi là một vấn đề lớn cần giải quyết. Dự đoán khách hàng rời bỏ có nghĩa là phát hiện khách hàng nào có khả năng rời khỏi dịch vụ hoặc hủy đăng ký dịch vụ. Khi bạn có thể xác định những khách hàng có nguy cơ hủy bỏ, bạn nên biết chính xác hành động tiếp thị nào cần thực hiện cho từng khách hàng riêng lẻ để tối đa hóa cơ hội khách hàng sẽ ở lại.
 <b>Nhiệm vụ vủa bạn là dựa trên tập dữ liệu đã có, xây dựng mô hình có thể dự đoán được khách hàng có khả năng rời bỏ trong tương lai</b>
* Mã nguồn gồm các file chính:
  - description.csv: file csv mô tả các trường trong tập dữ liệu
  - train.csv: tập huấn luyện sau khi được chia tách 
  - test.csv: tập kiểm tra sau khi được chia tách
  - split_data.py: file python dùng để chia tập dữ liệu gốc ra làm 2 file train.csv và test.csv, đồng thời cũng dùng để điền lại các giá trị không đồng nhất và giá trị khác biệt có trong tập dữ liệu
  - exp: thư mục hoạt động chính, bao gồm:
    - EDA.ipynb: file notebook cho việc khai phá dữ liệu
    - Datapipeline.py: file python tạo pipeline xử lý dữ liệu
    - OutlierHandling.py: file python xử lý outlier (viết theo api của scikit-learn)
    - CrossValidation.ipynb: file notebook thử nghiệm mô hình và lựa chọn, đánh giá bằng kiểm định chéo
    - Predict.ipynb: file notebook dự đoán lại trên tập kiểm tra
