from collections import Counter
import math, random
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

# 칼럼값으로 추가 - 함수 작성
# 1. cv_diff_value : 종가 일간 변화량
def cv_diff_value(prevalue, postvalue):
    return postvalue - prevalue


# 2. cv_diff_rate : 종가 일간 변화율
def cv_diff_rate(prevalue, postvalue):
    return (postvalue - prevalue) / prevalue * 100


# 3. cv_maN_value : 종가의 N일 이동평균
def cv_maN_value(cv, N):
    # min_period 옵션을 이용하여 할 수도 있음 // 데이터가 최소 x 개라도 존재하면 이동평균을 구함
    str_replay = "N의 값을 다시 입력해주세요"
    if 3 <= N <= 5:
        return cv.rolling(window=N).mean()
    else:
        return str_replay


# 4. cv_maN_rate : 종가의 N일 이동평균의 일간 변화율
def cv_maN_rate(cv, N):
    str_replay = "N의 값을 다시 입력해주세요"
    if 3 <= N <= 5:
        # DataFrame 을 list 로 변환
        for i in range(cv.index[0], (len(cv)+cv.index[0]), 1):
            cv_list.append(cv[i])
        # 종가의 N일 이동평균의 일간 변화율을 list 에 담기
        for i in range(len(cv_list)-1):
            if cv_list[i] != 0:
                cv_ma_rate.append((cv_list[i+1] - cv_list[i]) / cv_list[i] * 100)
            else:
                cv_ma_rate.append(0)
        # 종가의 N일 이동평균의 일간 변화율을 소수점 2째자리 까지 표현
        for i in range(len(cv_ma_rate)):
            cv_ma_rate_round.append(round(cv_ma_rate[i], 2))
        return cv_ma_rate_round
    else:
        return str_replay


# 5. ud_Nd : (a) N일 연속 증가  1, (b) N일 연속 하락 -1, (c) 그렇지 않은 날 0
def ud_Nd(cvdv, N):
    cvdv_list = [] # list
    un_Nd_list = [] # list
    # print(cvdv) # 종가
    # print(len(cvdv)) # 길이 : 230
    # DataFrame 을 list 로 변환
    for i in range(cvdv.index[0], (len(cvdv)+cvdv.index[0]), 1):
        cvdv_list.append(cvdv[i])
    # 알 수 없는 정보는 '0'으로 두겠다
    for i in range(N-2):
        un_Nd_list.append(0)
    # 상승, 하락, 그렇지 않은 날 계산
    for i in range(len(cvdv_list)-N+1): # 0 ~ 225
        increase_count = decrease_count = nothing_count = 0
        for j in range(N-1): # 0 ~ 3
            if cvdv_list[i + j] < cvdv_list[i + j + 1]: # 종가가 상승한 날
                increase_count += 1
            elif cvdv_list[i + j] > cvdv_list[i + j + 1]: # 종가가 하락한 날
                decrease_count += 1
            else: # 종가가 상승도 하락도 아닌날
                nothing_count += 1
        # N일 연속 종가가 상승, 하락, 그렇지 않은 날 판단하고 (N-1)날에 삽입
        if increase_count == (N - 1):
            un_Nd_list.append(1)
        elif decrease_count == (N - 1):
            un_Nd_list.append(-1)
        else:
            un_Nd_list.append(0)
    un_Nd_list.append(0) # 마지막날은 판단할 수 없어서 '0' 으로 삽입
    return un_Nd_list


# csv 파일 읽어오기 // DataFrame으로 변경 // NaN값 제거
csv_file_read = open('stock_history.csv', 'r', encoding='euc-kr')
stock_data = pd.read_csv(csv_file_read)
df = pd.DataFrame(stock_data)
stock_DataFrame = df.dropna(axis=1)

# 반복 시작
while True:
    # 초기값
    cv_amount = [0]  # 종가 일간 변화량을 저장할 list
    cv_rate = [0]  # 종가 일간 변화율을 저장할 list
    cv_ma_rate = [0]  # 종가의 N일 이동평균의 일간 변화율을 저장할 list
    un_Nd_plus = un_Nd_minus = 0  # 20회이상 판단할 count 변수
    result3 = []  # 종가의 N일 이동평균을 저장할 list
    result4 = []  # 종가 N일 이동평균의 일간 변화율
    cv_list = []  # 종가의 N일 이동평균의 일간 변화율을 저장할 list
    cv_ma_rate_round = []  # 종가의 N일 이동평균의 일간 변화율을 소수점 2자리로 저장할 list
    unNd_list = []  # 종가의 N일 증감을 저장할 list

    # 종목을 선택하고 N의 값을 입력받는다
    stock_name = input("종목을 입력해주세요 : ")
    Number = int(input("N의 값을 입력해주세요 : "))
    one_stock = stock_DataFrame.loc[ stock_DataFrame["stockname"] == stock_name]
    # print(one_stock)

    close_value = one_stock["close_value"] # 종가만 가져오기
    one_stock_copy = one_stock.copy() # DataFrame 에 열을 추가하기 위해 복사

    # print(close_value.index[0])
    # print(len(close_value)+clㅌose_value.index[0]-1)
    # print(len(close_value))
    # 종가 일간 변화량
    for i in range(close_value.index[0], (len(close_value)+close_value.index[0])-1, 1):
        result = cv_diff_value(close_value[i], close_value[i+1])
        cv_amount.append(result)
    one_stock_copy["cv_diff_value"] = cv_amount # DataFrame 에 데이터 추가
    # print(one_stock_copy)

    # 종가 일간 변화율 // 종가 일간 변화량과 마찬가지 // 소수점 2자리 표현
    for i in range(close_value.index[0], (len(close_value)+close_value.index[0])-1, 1):
        result2 = round(cv_diff_rate(close_value[i], close_value[i+1]), 2)
        cv_rate.append(result2)
    one_stock_copy["cv_diff_rate"] = cv_rate # DataFrame 에 데이터 추가
    # print(one_stock_copy)


    # 종가 N일 이동평균
    res3 = cv_maN_value(close_value, Number)
    if isinstance(res3, str):
        print(res3)
        continue
    else:
        result3 = res3.fillna(0)  # NaN값을 0으로 치환
        one_stock_copy["cv_maN_value"] = result3
    # print(one_stock_copy)

    # 종가 N일 이동평균의 일간 변화율
    ma_value = one_stock_copy["cv_maN_value"] # 종가 N일 이동평균 가져오기
    result4 = cv_maN_rate(ma_value, Number)
    if isinstance(result4, str):
        print(result4)
        continue
    else:
        one_stock_copy["cv_maN_rate"] = result4
    # print(one_stock_copy)

    # N일 연속 상승, 하락, 그렇지 않은 날 파악
    result5 = ud_Nd(close_value, Number)
    one_stock_copy["ud_Nd"] = result5
    # print(one_stock_copy)


    # un_Nd = 1, -1이 20회 이상 발생하도록 N을 3 ~ 5로 조정, 종목을 변경
    un_Nd_value = one_stock_copy["ud_Nd"] # N일 연속되는 증감 column 가져오기
    # print(un_Nd_value)
    # DataFrame 을 list 로 변환
    for i in range(un_Nd_value.index[0], (len(un_Nd_value)+un_Nd_value.index[0]), 1):
        unNd_list.append(un_Nd_value[i])
    # 20회 이상 발생하는지 판단
    for i in range(len(unNd_list)):
        if unNd_list[i] == 1:
            un_Nd_plus += 1
        if unNd_list[i] == -1:
            un_Nd_minus += 1

# 발생했다면 반복문을 종료하고 발생하지 않았다면 N을 조정하거나 종목을 변경한다
    if un_Nd_plus >= 20 and un_Nd_minus >= 20:
        print("un_Nd의 1 or -1 발생횟수가 둘 다 20을 넘지 않았습니다")
        continue
    else:
        break


# 추가 할 column 생성
raw_data = one_stock_copy[['cv_diff_value', 'cv_diff_rate', 'cv_maN_value', 'cv_maN_rate', 'ud_Nd']]
# 원본데이터에 column 을 추가
final_result = pd.concat([stock_DataFrame, raw_data], join='outer', axis=1, join_axes=None)

# 반복문이 끝나고 20회이상 발생하는 조건을 만족하면 csv파일(stock_history_added.csv)로 저장
final_result.to_csv('stock_history_added.csv', encoding='ms949')
print("Data가 성공적으로 추가됐습니다")
csv_file_read.close()
