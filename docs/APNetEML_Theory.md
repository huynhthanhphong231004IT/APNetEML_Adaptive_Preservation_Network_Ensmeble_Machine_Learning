
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
  <img src="../images/APNET_EML.png" width="800">
  <br>
  <i>Kiến trúc mô hình APNet-EML</i>
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
    <b>1. Hàm kích hoạt SN-ReLU (Smooth Nonlinear ReLU)</b>
  </span>
</h3>

<p align="justify">
SN-ReLU là hàm kích hoạt trơn, phi đơn điệu được tổng hợp và lấy cảm
hứng từ Swish và Softsign. “SN–ReLU đặc biệt phù hợp cho các tác vụ thị giác
yêu cầu kích hoạt phi đơn điệu mượt mà, lan truyền gradient ổn định và bảo toàn
các tín hiệu phân biệt yếu, chẳng hạn như nhận dạng chi tiết và phân tích hình
ảnh chất lượng thấp”. SN-ReLU là hàm mở rộng của ReLU ở miền âm và đặc tính trơn với các tính
chất: (i) Tính liên tục; (ii) Tính khả vi; (iii) tính bị chặn về tốc độ biến thiên;
(iv) Tính giảm các nơ-ron chết (miền âm của ReLU) và (v) Khắc phục biến mất
đạo hàm ở dạng kỳ vọng. Hàm SN–ReLU được định nghĩa bởi:
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?SN-ReLU(x)=x%5Csigma(0.8x)+%5Cfrac%7B0.1x%7D%7B1+%7Cx%7C%7D" width="300">
</p>

<p align="left">
  <span>Trong đó, sigmoid được định nghĩa là:</span>
  <img src="https://latex.codecogs.com/svg.image?%5Csigma(x)=%5Cfrac%7B1%7D%7B1+e%5E%7B-x%7D%7D" width="100" style="vertical-align: middle;">
</p>

<p align="center">
  <img src="../images/SNReLU.jpg" width="600">
  <br>
  <i>Minh họa hàm SN–ReLU (a) và đạo hàm của SN–ReLU (b) trên trục số thực.</i>
</p>

<p align="center">
  <img src="../images/DacTrungActicationConv1.png" width="900">
  <br>
  <i>So sánh đặc trưng (feature maps) giữa các hàm kích hoạt (ReLU, SNReLU, PReLU, ELU) tại 8 kênh (channels) đầu ra của  <mark><b>lớp Convolution 2D đầu tiên (Conv1)</b></mark> trên cùng một ảnh đầu vào là chiếc xe tăng.</i>
</p>

<p align="center">
  <img src="../images/DacTrungActicationConv6.png" width="900">
  <br>
  <i>So sánh đặc trưng (feature maps) giữa các hàm kích hoạt (ReLU, SNReLU, PReLU, ELU) tại 8 kênh (channels) đầu ra của <mark><b>lớp Convolution 2D cuối cùng (Conv6)</b></mark>  trên cùng một ảnh đầu vào là chiếc xe tăng.</i>
</p>

<h3 align="left">
  <span style="color:#8B4513;">
    <b>2. Hàm kích hoạt ANASP-ReLU (Antique-Aware Soft-Preserving Rectified Linear Unit)</b>
  </span>
</h3>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?f(x)%3D%20ANASP-ReLU(x)%3D%20%5Cbegin%7Bcases%7D%20%5Clambda%20x%2C%20%26%20%5Ctext%7Bif%20%7D%20x%20%5Cleq%200%2C%20%5C%5C%20%5Calpha%20x%5E%7B%5Cgamma%7D%2C%20%26%20%5Ctext%7Bif%20%7D%200%20%3C%20x%20%3C%20%5Ctau%2C%20%5C%5C%20%5Calpha%20%5Ctau%5E%7B%5Cgamma%7D%20%2B%20%5Cbeta%28x-%5Ctau%29%2C%20%26%20%5Ctext%7Bif%20%7D%20x%20%5Cgeq%20%5Ctau.%20%5Cend%7Bcases%7D" width="550">
</p>
<p align="justify">
Trong đó, không gian tham số được ràng buộc bởi các điều kiện hình học sau:
<img src="https://latex.codecogs.com/svg.image?%5Clambda%20%5Cin%20%280%2C1%29" width="45" style="vertical-align: middle;">
là hệ số rò rỉ âm (negative leakage coefficient), đóng vai trò giữ lại thành phần thông tin âm nhẹ.
<img src="https://latex.codecogs.com/svg.image?%5Calpha%2C%5Cbeta%20%5Cin%20%5Cmathbb%7BR%7D%5E%2B" width="80" style="vertical-align: middle;">
là các hệ số tỷ lệ dốc (scaling factors).
<img src="https://latex.codecogs.com/svg.image?%5Cgamma%20%5Cin%20%281%2C%2B%5Cinfty%29" width="95" style="vertical-align: middle;">
là tham số cấu trúc điều khiển bậc phi tuyến tại vùng kích hoạt yếu.
<img src="https://latex.codecogs.com/svg.image?%5Ctau%20%5Cin%20%5Cmathbb%7BR%7D%5E%2B" width="60" style="vertical-align: middle;">
đóng vai trò là ngưỡng chuyển pha hình học (phase transition threshold).
</p>

<p align="center">
  <img src="../images/ANASPReLU.jpg" width="600">
  <br>
  <i><p>
Biểu diễn hình học của hàm kích hoạt đề xuất dưới các cấu hình tham số thích nghi, tương ứng với các trạng thái phân hóa đặc trưng:
phi tuyến nhẹ (Đường xanh dương với
<img src="https://latex.codecogs.com/svg.image?%5Clambda%3D0.05%2C%5Calpha%3D0.70%2C%5Cgamma%3D1.3%2C%5Ctau%3D1.2%2C%5Cbeta%3D0.9" width="330" style="vertical-align: middle;">),
trạng thái cân bằng (Đường đỏ với
<img src="https://latex.codecogs.com/svg.image?%5Clambda%3D0.15%2C%5Calpha%3D0.90%2C%5Cgamma%3D1.8%2C%5Ctau%3D1.5%2C%5Cbeta%3D1.1" width="330" style="vertical-align: middle;">)
và phi tuyến mạnh (Đường xanh lá với
<img src="https://latex.codecogs.com/svg.image?%5Clambda%3D0.25%2C%5Calpha%3D1.1%2C%5Cgamma%3D2.5%2C%5Ctau%3D2%2C%5Cbeta%3D1.4" width="320" style="vertical-align: middle;">).
</p></i>
</p>

<p align="center">
  <img src="../images/DacTrungANASPReLU.png" width="900">
  <br>
  <i>So sánh mật độ biểu diễn đặc trưng trong không gian nhúng (Embedding Dimension) giữa ANASPReLU và các hàm kích hoạt ở tầng cuối trước khi đưa vào hàm mất mát trên cùng một ảnh đầu vào là chiếc xe tăng.</i>
</p>

<h3 align="left">
  <span style="color:#8B4513;">
    <b>3. Chiến lược hàm mất mát thích nghi động thích nghi ngữ cảnh PD - Loss (Proposed Dynamic Loss)</b>
  </span>
</h3>
<p align="justify">
  Nhằm tăng cường khả năng  <mark><b>phân tách liên lớp</b></mark> (inter-class separability), tối
  ưu độ <mark><b>co cụm nội lớp</b></mark> (intra-class compactness), đồng thời
  <mark><b>bảo toàn cấu trúc hình thái</b></mark> (morphological structures) đặc trưng của ảnh cổ vật, nghiên cứu này xây dựng
  một hàm mất mát tổng hợp đa thành phần (multi-component joint loss function).
  PD-Loss bao gồm các thành phần tối ưu hóa chuyên biệt tác động lên không
  gian nhúng đặc trưng (feature embedding space). Cơ chế cốt lõi của phương pháp
  là <mark><b>chiến lược học động trọng số</b></mark> (dynamic weight learning) giữa các thành phần
  loss. Để đảm bảo tính ổn định hội tụ và tránh hiện tượng sụp đổ cấu trúc đa
  tạp (manifold collapse), quá trình tối ưu được chia làm <mark><b>hai giai đoạn</b></mark>: Giai đoạn
  khởi tạo (warm-up phase) sử dụng hàm mất mát Cross-Entropy độc lập trong n
  epoch đầu tiên, và Giai đoạn tối ưu hóa thích nghi (adaptive optimization phase)
  cập nhật động các hệ số thông qua lan truyền ngược (backpropagation). Cụ thể,
  trọng số tổng hợp của mỗi thành phần loss bổ trợ được cấu thành từ tích của hai
  yếu tố: hằng số chuẩn hóa thang đo cố định βi và tham số trọng số học động λi.
  Hàm mất mát thích nghi PD-Loss được phát biểu một cách hình thức như sau
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cmathcal%7BL%7D_%7Btotal%7D%3D%5Cbegin%7Bcases%7D%5Calpha%5Cmathcal%7BL%7D_%7BCE%7D%2C%26%5Ctext%7Bif%20epochs%7D%5Cle%20n%2C%5C%5C%5B8pt%5D%5Calpha%5Cmathcal%7BL%7D_%7BCE%7D%2B%5Csum_%7Bi%3D1%7D%5E%7B4%7D%5Cleft%5Be%5E%7B-%5Clambda_i%7D%5Cleft%28%5Cbeta_i%2B%5Cgamma_i%5Cln%281%2Be%5E%7B%5Clambda_i%7D%29%5Cright%29%5Cmathcal%7BL%7D_i%2B%5Clambda_i%5Cright%5D%2C%26%5Ctext%7Bif%20epochs%7D%3E%20n.%5Cend%7Bcases%7D" width="650">
</p>
<p style="text-align: justify;">
Trong đó,
<img src="https://latex.codecogs.com/svg.image?%5Calpha%20%5Cin%20%5Cmathbb%7BR%7D%5E%2B" width="55" style="vertical-align: middle;">
là siêu tham số cố định đóng vai trò điều phối mức đóng góp của thành phần mất mát phân lớp chính;
<img src="https://latex.codecogs.com/svg.image?%5Cbeta_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%2B%2C%20i%3D1%2C%5Cdots%2C4" width="135" style="vertical-align: middle;">
là các hệ số chuẩn hóa tĩnh (static scaling coefficients) nhằm cân bằng thang đo (magnitude alignment) giữa các thành phần loss khác nhau trong không gian tối ưu hóa;
và
<img src="https://latex.codecogs.com/svg.image?%5Clambda_i%20%5Cin%20%5Cmathbb%7BR%7D%5E%2B" width="70" style="vertical-align: middle;">
là các trọng số học động (learnable dynamic weights) được cập nhật thông qua quá trình lan truyền ngược, cho phép mô hình tự thích nghi mức độ quan trọng của từng thành phần loss theo tiến trình huấn luyện.
Đồng thời, các thành phần
<img src="https://latex.codecogs.com/svg.image?L_i%20%28i%3D1%2C%5Cdots%2C4%29" width="100" style="vertical-align: middle;">
lần lượt được định nghĩa như sau:
<img src="https://latex.codecogs.com/svg.image?L_1%5Cequiv%20L_%7BICCL%7D" width="85" style="vertical-align: middle;">
(Intra-Class Compactness Loss) nhằm thu nhỏ phương sai nội lớp trong không gian đặc trưng;
<img src="https://latex.codecogs.com/svg.image?L_2%5Cequiv%20L_%7BICMRL%7D" width="90" style="vertical-align: middle;">
(Inter-Class Margin Ranking Loss) nhằm thiết lập biên phân tách an toàn giữa các lớp;
<img src="https://latex.codecogs.com/svg.image?L_3%5Cequiv%20L_%7BPSL%7D" width="70" style="vertical-align: middle;">
(Prototype Separation Loss) nhằm tối đa hóa khoảng cách hình học giữa các prototype của các lớp trong không gian nhúng
và
<img src="https://latex.codecogs.com/svg.image?L_4%5Cequiv%20L_%7BUAER%7D" width="75" style="vertical-align: middle;">
(Uncertainty-Aware Entropy Regularization) nhằm điều tiết mức độ bất định của mô hình, qua đó giảm ảnh hưởng của các mẫu nhiễu và các mẫu khó (hard examples) trong quá trình tối ưu hóa.
</p>
<p style="text-align: justify;">

<b>Cơ sở lý thuyết của giai đoạn tiền ổn định với n epoch đầu:</b>
Việc phân tách quy trình huấn luyện thành cấu trúc hai giai đoạn xuất phát từ các đặc điểm hình học vi mô và động lực học gradient (gradient dynamics) của mạng nơ-ron sâu:

<i>(1) Định hình cấu trúc đa tạp sơ bộ.</i>
Tại thời điểm khởi đầu, các trọng số của mạng nền tảng (backbone network) nằm ở trạng thái ngẫu nhiên, chưa tương thích với phân phối dữ liệu cổ vật (P<sub>data</sub>). Hàm L<sub>CE</sub> đóng vai trò như một cơ chế định hướng thô (coarse alignment), ép mô hình tập trung khai thác các đặc trưng biểu diễn mức thấp (low-level representations) như kết cấu cục bộ và phân phối màu sắc. Quá trình này giúp thiết lập một cấu trúc phân lớp topo sơ bộ trên đa tạp đặc trưng trước khi áp dụng các ràng buộc hình học nghiêm ngặt;

<i>(2) Ngăn ngừa sụp đổ không gian nhúng và bất ổn định Gradient.</i>
Nếu các ràng buộc hình học phức tạp (L<sub>ICCL</sub>, L<sub>ICMRL</sub>, L<sub>PSL</sub>) cùng cơ chế cập nhật động λ<sub>i</sub> được kích hoạt đồng thời ngay từ epoch đầu tiên, mô hình rất dễ rơi vào các điểm tối ưu cục bộ kém (poor local minima) do không gian vector chưa được định hình. Hơn nữa, việc thiếu một “hướng neo” vững chắc từ L<sub>CE</sub> sẽ khiến các tham số động λ<sub>i</sub> dao động hỗn loạn, dẫn đến hiện tượng bùng nổ hoặc tiêu biến gradient (gradient explosion/vanishing). Do đó, việc cố định n epoch đầu tạo ra một vùng đệm hội tụ ổn định, chuẩn bị một không gian nhúng có độ chín muồi thích hợp cho giai đoạn tối ưu hóa đa mục tiêu kế tiếp.

<b>Động cơ toán học của hàm mất mát tích hợp đa thành phần:</b>
Dữ liệu ảnh cổ vật sở hữu các thuộc tính đặc thù gây bất lợi cho các hàm mất mát truyền thống: tỷ lệ tín hiệu trên nhiễu (SNR) thấp, tổn thương hình thái do dòng thời gian, và hiện tượng độ tương đồng liên lớp cao kết hợp với biến động nội lớp lớn (high inter-class similarity and intra-class variance). Một hàm loss Cross-Entropy đơn lẻ chỉ tập trung vào ranh giới quyết định (decision boundary) tại lớp ngoài cùng mà hoàn toàn bỏ qua cấu trúc hình học bên trong của không gian nhúng. Do đó, hàm mục tiêu tổng thể L<sub>total</sub> được đề xuất không thuần túy là một phép tổ hợp tuyến tính (linear combination), mà cấu thành một hệ thống tối ưu hóa đa tầng ràng buộc:

<i>(1) Nhóm ràng buộc topo không gian nhúng (L<sub>1,2,3</sub>).</i>
Thiết lập một cấu trúc hình học lý tưởng theo triết lý "tối đa khoảng cách giữa các lớp, tối thiểu khoảng cách trong cùng một lớp" dưới dạng phân rã các cụm siêu cầu (hyperspherical clustering);

<i>(2) Nhóm bảo toàn cấu trúc thị giác (L<sub>4</sub>).</i>
Đóng vai trò như một bộ lọc thông thấp (regularizer) giữ lại các thông tin bất biến về mặt hình thái học của cổ vật, tránh hiện tượng mô hình bị mất các đặc trưng ngữ cảnh tinh vi trong quá trình nén chiều đặc trưng;

<i>(3) Nhóm điều tiết phân phối xác suất (L<sub>5</sub>).</i>
Đóng vai trò kiểm soát entropy cấu trúc, làm mượt ranh giới quyết định đối với các mẫu nằm ở vùng bất định cao, tăng cường năng lực tổng quát hóa (generalization capability) của toàn bộ kiến trúc trên các tập dữ liệu thực tế độc lập.

</p>

