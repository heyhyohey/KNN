import pandas as pd

# cv_diff_value(종가 일간 변화량)

# cv_diff_rate(종가 일간 변화율)

# cv_maN_value(종가의 N일 이동평균)

# cv_maN_rate(종가의 N일 이동평균의 일반 변화율)

# ud_Nd(N일 연속 종가 상승, 하락 수치)

if __name__ == "__main__":
    # 1. stock_history.csv 파일 읽기
    stock_history = pd.read_csv("stock_history.csv", encoding="euc-kr")

    # 2. 비어있는 column 삭제
    stock_history = stock_history.drop(['Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13'], 1)
    print(stock_history)
