import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# تحميل الداتا سيت
# =========================

dataset = pd.read_csv("dataset.csv")

# =========================
# عدد الصفوف والأعمدة
# =========================

print("\n========== DATASET INFO ==========")
print(f"Rows (عدد الصفوف): {dataset.shape[0]}")
print(f"Columns (عدد الأعمدة): {dataset.shape[1]}")

# =========================
# التحقق من أن الداتا فوق 100 ألف
# =========================

if dataset.shape[0] >= 100000:
    print("✅ Dataset size is VALID (100k+)")
else:
    print("❌ Dataset size is TOO SMALL")

# =========================
# تقسيم البيانات
# =========================

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# بناء النموذج
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# التوقع
# =========================

predictions = model.predict(X_test)

# =========================
# حساب الدقة
# =========================

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULT ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

# =========================
# التحقق من الدقة المطلوبة
# =========================

if 0.83 <= accuracy <= 0.98:
    print("✅ Accuracy is within required range")
else:
    print("⚠ Accuracy is outside required range")

print("\n✅ Model is working successfully")