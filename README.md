<h2 align="center">
  <span style="color:#8B4513;">
    <b>Mô hình phân loại 2 giai đoạn APNet-EML (Adaptive Preservation Network - Ensemble Machine Learning)</b>
  </span>
</h2>

<p align="justify">
<i>
Mô hình APNet-EML gồm 2 giai đoạn: Giai đoạn 1 trích xuất đặc trưng sâu qua APNetCNN được xây dựng dựa trên sự bảo toàn lưu lượng thông tin qua các tầng Conv bằng 2 hàm kích hoạt đề xuất là SNReLU (Smooth Nonlinear ReLU), ANASPReLU (Antique-Aware Soft-Preserving Rectified Linear Unit), kết hợp cùng hàm mất mát đa giai đoạn kích hoạt PD-Loss (Proposed Dynamic Loss) dựa trên Cross Entropy Loss, Artifact Structure Preservation Loss, Inter-Class Margin Ranking Loss, Intra-Class Compactness Loss, Prototype Separation Loss, Uncertainty-Aware Entropy Regularization. Nhờ ưu điểm của hàm kích hoạt và quá trình phân tách đặc trưng tối ưu của hàm mất mát nên mô hình giảm được các tầng Full Connection dày đặc vốn là đặc trưng của các mô hình truyền thống, do đó giảm số lượng parameters hơn 90% so với ResNet50. Giai đoạn 2 phân loại tập thể được thiết kế dựa trên sự suy giảm bias từ 4 họ mô hình máy học truyền thống SVM, Random Forest, Logistic Regression và MLP. Kết quả cho thấy mô hình đạt độ chính xác gần 90% trên tập CIFAR-10 với chưa tới 1M tham số.
</i>
</p>
<p align="center">
  <img src="images/APNET_EML.png" width="800">
  <br>
  <i>Hình 1. Kiến trúc mô hình APNet-EML</i>
</p>
<p style="text-align: justify;">

<b>A. Cấu trúc APNetCNN - trích xuất đặc trưng sâu</b>

Chúng tôi đề xuất một kiến trúc mạng nơ-ron tích chập tinh gọn được tối ưu hóa cho nhiệm vụ trích xuất và biến đổi không gian nhúng biểu diễn. Cơ chế hoạt động bao gồm ba giai đoạn xử lý chính: (1) Ảnh đầu vào sau bước tăng cường dữ liệu được đưa qua ba khối Conv2D nông với số kênh lần lượt 32 → 64 → 128 nhằm học các đặc trưng hình học cơ sở như biên, góc và kết cấu; (2) Bản đồ đặc trưng hai chiều được nén trực tiếp thành vectơ một chiều thông qua tầng Global Average Pooling (GAP), loại bỏ hoàn toàn các tầng Fully Connected truyền thống vốn chiếm phần lớn số lượng tham số trong mạng CNN; (3) Vectơ đặc trưng tiếp tục được tinh chỉnh qua hai tầng Dense gồm 256 nút để sinh ra vectơ nhúng biểu diễn đặc trưng có khả năng phân tách hình học tối đa z ∈ R<sup>256</sup>.

Bản chất của việc phân bổ phân cấp hai hàm kích hoạt phi tuyến SN-ReLU và ANASP-ReLU xuất phát từ sự khác biệt về vai trò toán học và chi phí tính toán tại từng độ sâu của kiến trúc mạng:

<i>(1) SN-ReLU tại các tầng trích xuất đặc trưng đầu:</i> Các tầng Conv2D nông xử lý bản đồ đặc trưng hai chiều có kích thước lớn (H × W × C). SN-ReLU kết hợp đặc tính phi tuyến trơn với thành phần rò rỉ âm nhỏ, giúp duy trì luồng đạo hàm liên tục trong quá trình lan truyền ngược, từ đó hạn chế hiện tượng mất gradient khi học các đặc trưng hình học mức thấp.

<i>(2) ANASP-ReLU tại tầng Dense 256 cuối:</i> Sau khi bản đồ đặc trưng được nén thành vectơ nhúng 256 chiều, ANASP-ReLU được sử dụng để tinh chỉnh không gian biểu diễn. Với ba vùng kích hoạt riêng biệt (miền âm nhẹ, miền phi tuyến và miền tuyến tính), hàm kích hoạt này đóng vai trò như một cơ chế biến đổi manifold thích nghi, giúp nén các chiều chứa thông tin dư thừa, đồng thời mở rộng khoảng cách giữa các đặc trưng mang tính phân biệt cao, từ đó nâng cao khả năng phân tách của vectơ nhúng trước giai đoạn phân lớp.

Để tăng cường khả năng phân tách của không gian nhúng 256 chiều, kiến trúc áp dụng cơ chế huấn luyện hai giai đoạn thông qua PD-Loss. Cơ chế này giúp thu hẹp khoảng cách giữa các mẫu cùng lớp quanh tâm đại diện (Prototype), đồng thời mở rộng khoảng cách giữa các lớp khác nhau và giảm ảnh hưởng của các mẫu khó (hard examples), từ đó hình thành không gian nhúng có tính phân biệt cao.

<b>B. Cấu trúc phân loại tập thể EML</b>

Tầng phân loại cuối được thiết kế theo cơ chế học tập thể (Ensemble Learning) dựa trên chiến lược Soft Voting bao gồm bốn bộ phân loại học máy gồm Support Vector Machine (SVM), Random Forest (RF), Logistic Regression (LogReg) và Multi-Layer Perceptron (MLP). Thiết kế này không phát sinh quá trình lan truyền ngược tại giai đoạn phân loại, qua đó duy trì tổng số tham số huấn luyện của toàn bộ mô hình ở mức tối ưu.

Ở cơ chế phân loại cuối Ensemble này có kết hợp cơ chế Stacking hai tầng giữa Random Forest và Logistic Regression, trong đó RF đóng vai trò bộ học cơ sở tạo ra các phân phối xác suất ban đầu, còn LogReg thực hiện hiệu chỉnh xác suất trước khi đưa vào cơ chế bỏ phiếu mềm. Nhờ vậy, các ranh giới quyết định dạng phân mảnh đặc trưng của tập hợp cây được làm mượt, giảm hiện tượng dự đoán quá tự tin, đồng thời hạn chế nguy cơ quá khớp và cải thiện khả năng tổng quát hóa trên dữ liệu chưa quan sát.

Về bản chất toán học, bốn bộ phân loại đại diện cho bốn nguyên lý tối ưu hóa khác nhau: SVM tối đa hóa khoảng cách biên, RF học các quan hệ phi tuyến thông qua tập hợp cây quyết định, LogReg mô hình hóa xác suất hậu nghiệm bằng hàm logistic, trong khi MLP khai thác các phép biến đổi phi tuyến nhiều tầng. Sự kết hợp này tạo nên tính đa dạng cao của ranh giới quyết định, giúp các sai số riêng lẻ của từng mô hình được bù trừ lẫn nhau thay vì cộng dồn.

Trên nền không gian nhúng 256 chiều đã được tối ưu bởi tầng rút trích đặc trưng sâu ban đầu, cơ chế Soft Voting tổng hợp phân phối xác suất của bốn bộ phân loại để hình thành quyết định cuối cùng có độ tin cậy cao hơn, ổn định hơn và ít nhạy cảm với nhiễu hoặc sự mất cân bằng dữ liệu.

</p>

<h3 align="left">
  <span style="color:#8B4513;">
    <b>Hàm kích hoạt SN-ReLU (Smooth Nonlinear ReLU)</b>
  </span>
</h3>

<math block value="SN\\text{-}ReLU(x)=x\\sigma(0.8x)+\\frac{0.1x}{1+|x|}"/>


<p align="center">
  <img src="images/SNReLU.jpg" width="800">
  <br>
  <i>Hình 2. Minh họa hàm SN–ReLU (a) và đạo hàm của SN–ReLU (b) trên trục số thực.</i>
</p>


