import cv2
import pickle

# 1. Load mô hình đã huấn luyện
with open('knn_model.pkl', 'rb') as model_file:
    knn, le, scaler = pickle.load(model_file)

# 2. Hàm dự đoán màu sắc
def predict_color(r, g, b):
    # Chuẩn hóa giá trị đầu vào
    scaled_values = scaler.transform([[r, g, b]])
    # Thực hiện dự đoán
    color_index = knn.predict(scaled_values)[0]
    color_name = le.inverse_transform([color_index])[0]
    return color_name

# 3. Nhận diện màu qua camera
def detect_color_from_camera():
    cap = cv2.VideoCapture(0)  # Mở camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lấy kích thước ảnh và màu trung tâm
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        b, g, r = frame[center_y, center_x]
        color_name = predict_color(r, g, b)

        # Hiển thị màu trên màn hình
        cv2.putText(frame, f"Color: {color_name} (R: {r}, G: {g}, B: {b})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 10, (int(b), int(g), int(r)), -1)
        cv2.imshow('Color Detection', frame)

        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy chương trình
if __name__ == "__main__":
    detect_color_from_camera()
