import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

################# 2. K-NN ###################

# column 이 추가된 csv 파일 읽어오기 // DataFrame으로 변경 // NaN값 제거
csv_add_file_read = open('stock_history_added.csv', 'r', encoding='euc-kr')
stock_data_add = pd.read_csv(csv_add_file_read)
df = pd.DataFrame(stock_data_add)
# stock_DataFrame_add = df.dropna(axis=1)

one_stock_add = df.loc[ df["stockname"] == "LG이노텍"]
# print(one_stock_add)

# print(one_stock_add[["cv_diff_value", "cv_diff_rate"]]) # 독립변수
# print(one_stock_add["ud_Nd"]) # 종속변수

"""
# 1. 일간 종가 변화량, 일간 종가 변화율
X = one_stock_add[["cv_diff_value", "cv_diff_rate"]] # 독립변수
y = one_stock_add["ud_Nd"] # 종속변수
"""

X = one_stock_add[["cv_diff_value", "cv_maN_value"]] # 독립변수
y = one_stock_add["ud_Nd"] # 종속변수

# 학습데이터 : 검증데이터 = 7 : 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


accuracy_dictionary = dict() # K값에 따른 정확도를 저장할 dictionary
predict_column = dict() # K값에 따른 예측값이 담긴 dictionary

# K값을 변경 // 0 입력시 종료
while True:
    K_Number = int(input("K값을 입력해주세요 : "))
    if K_Number == 0:
        break
    knn_model = KNeighborsClassifier(n_neighbors=K_Number, p=2, metric='minkowski')
    knn_model.fit(X_train, y_train) # 모델 생성 = 학습
    predict_column[K_Number] = knn_model.predict(X_test)
    accuracy = knn_model.score(X_test, y_test) # 정확도
    print(predict_column)
    print(accuracy)
    accuracy_dictionary[K_Number] = accuracy # dictionary에 데이터 추가

print(predict_column)
print(accuracy_dictionary)

# for value in accuracy_dictionary.values():
