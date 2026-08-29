import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def run_ensemble_pipeline(model, train_data, val_data=None, batch_size=64, pca_components=256, output_dir="./Model"):
    print(" >>> [Pipeline] Bắt đầu tự động trích xuất đặc trưng & Ensemble")
    if isinstance(train_data, tuple):
        X_tr, y_tr = train_data
        if isinstance(val_data, tuple):
            X_va, y_va = val_data
            X_all = np.concatenate([X_tr, X_va], axis=0)
            y_all = np.concatenate([y_tr, y_va], axis=0)
        else:
            X_all, y_all = X_tr, y_tr
    else:
        print(" [Lỗi] train_data cần phải là Tuple (X, y) để chạy tự động Ensemble.")
        return
    print(" >>> [Step 1/3] Trích xuất đặc trưng")
    all_features = []
    for i in range(0, len(X_all), batch_size):
        batch_x = X_all[i:i + batch_size]
        feat_vec, _ = model.predict(batch_x, verbose=0)
        all_features.append(feat_vec)

    all_features = np.vstack(all_features).astype(np.float32)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "features_256.npy"), all_features)
    np.save(os.path.join(output_dir, "labels.npy"), y_all)
    print(f" >>> [Step 2/3] Chia tập dữ liệu & Giảm chiều PCA ({pca_components})...")
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        all_features,
        y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    n_comp = min(pca_components, X_train_f.shape[1])
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_f)
    X_val_pca = pca.transform(X_val_f)

    joblib.dump(pca, os.path.join(output_dir, "pca_256.pkl"))
    print(" >>> [Step 3/3] Huấn luyện Ensemble (Stacking + MLP + SVM)")
    lvl1 = [("rf", RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42))]
    lvl2 = StackingClassifier(
        estimators=lvl1,
        final_estimator=LogisticRegression(max_iter=500),
        passthrough=True,
        n_jobs=-1
    )
    ensemble = VotingClassifier(
        estimators=[
            ("stack1", lvl2),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42)),
            ("svm", SVC(probability=True, kernel="linear"))
        ],
        voting="soft",
        n_jobs=-1
    )
    ensemble.fit(X_train_pca, y_train_f)
    val_acc = ensemble.score(X_val_pca, y_val_f)
    print(f"\n [ Validation Accuracy ] Độ chính xác Ensemble trên tập Val: {val_acc * 100:.2f}%")
    joblib.dump(ensemble, os.path.join(output_dir, "ensemble_feature.pkl"))
    print(f" HOÀN THÀNH TOÀN BỘ PIPELINE! Mô hình Ensemble đã được lưu tại: {output_dir}")