<h2 align="center">
  <span style="color:#8B4513;">
    <b>Mô hình phân loại 2 giai đoạn APNet-EML (Adaptive Preservation Network - Ensemble Machine Learning)</b>
  </span>
</h2>

<p align="justify">
<i>
Mô hình APNet-EML gồm 2 giai đoạn: Giai đoạn 1 trích xuất đặc trưng sâu qua APNetCNN được xây dựng dựa trên sự bảo toàn lưu lượng thông tin qua các tầng Conv bằng 2 hàm kích hoạt đề xuất là SNReLU (Smooth Nonlinear ReLU), ANASPReLU (Antique-Aware Soft-Preserving Rectified Linear Unit), kết hợp cùng hàm mất mát đa giai đoạn kích hoạt PD-Loss (Proposed Dynamic Loss) dựa trên Cross Entropy Loss, Artifact Structure Preservation Loss, Inter-Class Margin Ranking Loss, Intra-Class Compactness Loss, Prototype Separation Loss, Uncertainty-Aware Entropy Regularization. Nhờ ưu điểm của hàm kích hoạt và quá trình phân tách đặc trưng tối ưu của hàm mất mát nên mô hình giảm được các tầng Full Connection dày đặc vốn là đặc trưng của các mô hình truyền thống, do đó giảm số lượng parameters hơn 90% so với ResNet50. Giai đoạn 2 phân loại tập thể được thiết kế dựa trên sự suy giảm bias từ 4 mô hình máy học truyền thống SVM, Random Forest, Logistic Regression và MLP. Kết quả cho thấy mô hình đạt độ chính xác gần 90% trên tập CIFAR-10 với chưa tới 1M tham số.
</i>
</p>
<p align="center">
  <img src="images/APNET_EML.png" width="800">
  <br>
  <i>Hình 1. Kiến trúc mô hình APNet-EML</i>
</p>
