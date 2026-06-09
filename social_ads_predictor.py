# ================================================================
# مشروع: التنبؤ بشراء المنتج عبر الإعلانات الاجتماعية
# محاكي لـ Lab4 - Logistic Regression - 120,000 سجل
# ================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_csv("dataset.csv")
# --- 1. توليد البيانات (120,000 عميل) ---
np.random.seed(42)
n = 120000
age         = np.random.randint(18, 60, n)
salary      = np.random.randint(15000, 150000, n)
study_hours = np.random.uniform(0, 10, n)
internet    = np.random.choice([0, 1], n)
experience  = np.random.randint(0, 20, n)
score       = (salary/150000)*40 + study_hours*3 + internet*5 + experience*1.5 + np.random.normal(0, 8, n)
purchased   = (score > 32).astype(int)
df = pd.DataFrame({'age':age, 'salary':salary, 'study_hours':np.round(study_hours,2),
                   'internet_access':internet, 'experience':experience, 'purchased':purchased})

print("=" * 50)
print("  مشروع التنبؤ بشراء المنتج - Logistic Regression")
print("=" * 50)
print(f"[1] حجم البيانات    : {len(df):,} سجل")
print(f"    القيم المفقودة  : {df.isnull().sum().sum()}")
print(f"    سيشتري          : {df['purchased'].sum():,} ({df['purchased'].mean()*100:.1f}%)")
print(f"    لن يشتري        : {(1-df['purchased']).sum():,} ({(1-df['purchased'].mean())*100:.1f}%)")

# --- 2. تحضير + تقسيم + تطبيع ---
X = df[['age','salary','study_hours','internet_access','experience']].values
y = df['purchased'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
print(f"\n[2] بيانات التدريب  : {len(X_train):,} | بيانات الاختبار: {len(X_test):,}")

# --- 3. بناء وتدريب النموذج ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\n[3] تم تدريب النموذج بنجاح")

# --- 4. التقييم ---
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)
print(f"\n[4] دقة النموذج (Accuracy): {acc*100:.2f}%")
print("\n    تقرير التصنيف:")
print(classification_report(y_test, y_pred, target_names=['لن يشتري','سيشتري']))

print("\n" + "=" * 50)
print(f"  اكتمل المشروع | الدقة: {acc*100:.2f}%")
print("=" * 50)
