# machine learning workflow dengan scikit lear

# 1. persiapan dataset

  # load sample dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# 2. splitting dataset : training & testing test (membagi dataset ke dua bagian)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# 3. training model
  # pada scikit learn, model machine learning dibentuk dari class yang di kenal dengan istilah estimator
  # setiap estimator akan mengimplementasikan dua method utama, yaitu fit() dan predict()
  # method fit() digunakan untuk melakukan training model
  # method predict() digunakan untuk melakukan estimasi/prediksi dengan memanfaatkan trained model

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. evaluasi model
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'accuracy : {acc}')
print()

# 5. pemanfaatan trained model

  # seteleh model di training dan di tes evaluasi, kita akan coba gunakan untuk memprediksi data baru

data_baru = [[5, 5, 3, 2], [2, 4, 3, 5]]
preds = model.predict(data_baru)
print(preds)

pred_species = [iris.target_names[p] for p in preds]
print(f'hasil prediksi : {pred_species}')

# 6. trained model (model yang sudah di train) yang sudah siap ini pastinya kita akan deploy ke production
  # dump & load trained model

  # dumping model machine learning menjadi file joblib
import joblib

joblib.dump(model, 'iris_classifier_knn.joblib')

# 7. loading model machine learning dari file joblib
production_model = joblib.load('iris_classifier_knn.joblib')
