from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


################# 2. K-NN ###################

# column 이 추가된 csv 파일 읽어오기 // DataFrame으로 변경 // NaN값 제거
csv_add_file_read = open('stock_history_added.csv', 'r', encoding='euc-kr')
stock_data_add = pd.read_csv(csv_add_file_read)
df = pd.DataFrame(stock_data_add)

# 1. 일간 종가 변화량, 일간 종가 변화율
X = df[["cv_diff_value", "cv_diff_rate"]] # 독립변수
y = df["ud_Nd"] # 종속변수

# 학습데이터 : 검증데이터 = 7 : 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
df_copy = X_test.copy()

accuracy_dictionary = dict() # K값에 따른 정확도를 저장할 dictionary
predict_column = dict() # K값에 따른 예측값이 담긴 dictionary

K_Number_Final = 0

# K값을 변경 // 0 입력시 종료
while True:
    K_Number = int(input("K값을 입력해주세요 : "))
    if K_Number == 0:
        break
    K_Number_Final = K_Number
    knn_model = KNeighborsClassifier(n_neighbors=K_Number, p=2, metric='minkowski')
    knn_model.fit(X_train, y_train) # 모델 생성 = 학습
    predict_column[K_Number] = knn_model.predict(X_test)
    accuracy = knn_model.score(X_test, y_test) # 정확도
    print(predict_column[K_Number])
    print(accuracy)
    accuracy_dictionary[K_Number] = accuracy # dictionary에 데이터 추가

# 추가할 column 생성
df_copy["ud_Nd_predicted"] = predict_column[K_Number_Final]
raw_data = df_copy[['ud_Nd_predicted']]

# 원본데이터에 column 을 추가
final_result = pd.concat([df, raw_data], join='outer', axis=1, join_axes=None)

final_result.to_csv('stock_history_K.csv', encoding='ms949', index=False)
print("Data가 성공적으로 추가됐습니다")
csv_add_file_read.close()
