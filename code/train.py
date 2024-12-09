import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

# Tải và tiền xử lý dữ liệu màu sắc
def load_and_preprocess_data(filepath, sample_size=None):
    print("Loading data...")
    data = pd.read_csv(filepath)
    
    # Tối ưu kiểu dữ liệu
    data['r'] = data['r'].astype('uint8')
    data['g'] = data['g'].astype('uint8')
    data['b'] = data['b'].astype('uint8')
    data['colorname'] = data['colorname'].astype('category')
    
    # Lấy mẫu nếu được chỉ định
    if sample_size is not None:
        sample_size = min(sample_size, len(data))
        data = data.sample(n=sample_size, random_state=42)
    
    print(f"Dataset size: {len(data)} rows")
    
    # Chuẩn bị features và target
    X = data[['r', 'g', 'b']].values
    y = data['colorname']
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Mã hóa nhãn
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, scaler, le

# Huấn luyện và đánh giá mô hình phân loại màu sắc
def train_and_evaluate_model(model, X, y, test_size=0.2):
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Huấn luyện mô hình
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Đánh giá
    print("Evaluating model...")
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    return model

def save_model(model, le, scaler, filepath):
    # Lưu mô hình và các bộ chuyển đổi
    with open(filepath, 'wb') as f:
        pickle.dump((model, le, scaler), f)
    print(f"Model saved at '{filepath}'")

# Mã nguồn cho huấn luyện mô hình SVM:
"""
from sklearn.svm import SVC

X, y, scaler, le = load_and_preprocess_data('XKCDcolors_balanced.csv')
svm = SVC(kernel='linear', probability=True, class_weight='balanced')
svm = train_and_evaluate_model(svm, X, y)
save_model(svm, le, scaler, 'svm_model.pkl')
"""

# Mã nguồn cho huấn luyện mô hình KNN:
"""
from sklearn.neighbors import KNeighborsClassifier

X, y, scaler, le = load_and_preprocess_data('XKCDcolors_balanced.csv')
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn = train_and_evaluate_model(knn, X, y)
save_model(knn, le, scaler, 'knn_model.pkl')
"""

# Mã nguồn cho huấn luyện mô hình Decision Tree:
"""
from sklearn.tree import DecisionTreeClassifier

X, y, scaler, le = load_and_preprocess_data('XKCDcolors_balanced.csv')
tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
tree = train_and_evaluate_model(tree, X, y)
save_model(tree, le, scaler, 'decision_tree_model.pkl')
"""