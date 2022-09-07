import pandas as pd
import xgboost
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def preprocess2():
    df = pd.read_excel('./FHK2022_dataA.xlsx')

    delete_list = ['Work Time', 'Serial No.', 'Material Date', 'SHANK RUN OUT R1', 'BOWL 경도(HRc) 심부', 'Part Name']
    df = df.drop(labels=delete_list, axis=1)

    # non_unique 삭제

    non_unique = list()
    for col in df.columns:
        if df[col].nunique() == 1:
            non_unique.append(col)

    """non_unique
    # ['Outer Quenching Delay B', '1st step Moving velocity B', '2nd step Moving velocity B',
    # '2nd step Dure Time B', '2nd step Power Setting(%) B', '3rd step Moving velocity B', '3rd step Dure Time B',
    # '3rd step Power Setting(%) B', '4th step Moving velocity B', '4th step Power Setting(%) B', '1st step Moving velocity S',
    # '2nd step Position S', '2nd step Moving velocity S', '2nd step Dure Time S', '2nd step Power Setting(%) S', '3rd step Position S',
    # '3rd step Moving velocity S', '3rd step Power Setting(%) S', '4th step Power Setting(%) S', 'Quenching Density', 'Surface Hardness',
    # 'Cold Forging microstructure', 'Cold Forging grain size', 'Normalizing temperature', 'Normalizing time', 'SHANK RUN OUT R1']"""


    # Quenching Density 전처리
    # Quenching Density의 경우 데이터가 존재하는 경우 값이 전부 같기 때문에 최빈값, median 값이 의미가 없다.

    """ Quenching Density를 One-hot encoding 했을 때"""
    # df['Quenching Density One'] = 0
    #     # df.loc[(df['Quenching Density'] == 7.8, 'Quenching Density One')] = 1
    """ Quenching Density 결측치를 0으로 채웠을 때"""
    df['Quenching Density'] = df['Quenching Density'].fillna(0)
    """ Quenching Density 결측치를 7.8으로 채웠을 때"""
    # df['Quenching Density'] = df['Quenching Density'].fillna(7.8)

    # Quenching Density 전처리
    # Quenching Density의 경우 데이터가 존재하는 경우 값이 전부 같기 때문에 최빈값, median 값이 의미가 없다.

    """ Surface Hardness를 One-hot encoding 했을 때"""
    df['Surface Hardness One'] = 0
    df.loc[(df['Surface Hardness'] == 7.8, 'Surface Hardness')] = 1
    """ Surface Hardness 결측치를 0으로 채웠을 때"""
    # df['Surface Hardness'] = df['Surface Hardness'].fillna(0)
    """ Surface Hardness 결측치를 60으로 채웠을 때"""
    # df['Surface Hardness'] = df['Surface Hardness'].fillna(60)

    # df = df.drop(labels=['Quenching Density', 'Surface Hardness'], axis=1)
    df = df.drop(labels=['Surface Hardness'], axis=1)
    # df = df.drop(labels=['Quenching Density'], axis=1)


    # Material  Heat No. 전처리
    # unique value의 개수는 41개이다. one-hot encoding을 하기에는 너무 많은 숫자이지만
    # 문자열 형태이고 용광로 번호는 분명 경화 깊이에 영향을 끼칠 것이라고 예상한다.
    # 그러나 numerical 값을 넣게 된다면 데이터에 order를 부여하게 될 것이다. 물론, 용광로 번호는 성능과 상관 없을 것이라고 가정한다.

    """용광로 번호를 One-hot encoding 했을 때"""
    df['Material  Heat No.'] = df['Material  Heat No.'].fillna('non_value')
    for i in range(df['Material  Heat No.'].nunique()):
        df['Material Heat No.'+str(i)] = 0
        df.loc[(df['Material  Heat No.'] == df['Material  Heat No.'].unique()[i], 'Material Heat No.'+str(i))] = 1
    df = df.drop(labels='Material  Heat No.', axis=1)

    # C(%, X100) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'C(%, X100)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # Si(%, X100) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'Si(%, X100)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # Mn(%, X100) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'Mn(%, X100)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # P(%, X100) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'P(%, X100)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # S(%, X100) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'S(%, X100)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # Cu(%, X100) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'Cu(%, X100)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # Ni(%, X100) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'Ni(%, X100)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # Cr(%, X100) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'Cr(%, X100)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # Mo(%, X100) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'Mo(%, X100)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # Forging temperature(˚) 전처리
    # 결측값이 너무 많다. 버리기엔 아깝고 결측값을 채워 넣기에는 너무 신뢰할 수 없다.
    col_name = 'Forging temperature(˚)'
    """결측치를 0으로 대입했을 때"""
    df[col_name] = df[col_name].fillna(0)
    """결측치를 mean 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.mean(df[col_name]))
    """결측치를 median 값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(np.median(df[col_name]))
    """결측치를 최빈값으로 대입했을 때"""
    # df[col_name] = df[col_name].fillna(df[col_name].mode()[0])

    # Cold Forging Hardness 전처리
    # 값이 적어도 너무 적다.
    """결측치를 0으로 대입했을 때"""
    df['Cold Forging Hardness']= df['Cold Forging Hardness'].fillna(0)
    """결측치를 235으로 대입했을 때"""
    # df['Cold Forging Hardness']= df['Cold Forging Hardness'].fillna(235)
    """결측치를 mean 값으로 대입했을 때"""
    # df['Cold Forging Hardness']= df['Cold Forging Hardness'].fillna(np.mean(df['Cold Forging Hardness']))

    # Cold Forging microstructure 전처리
    # 이 과정에 대한 데이터는 거의 없다 싶이 한다.
    """Cold Forging microstructure 을 One-Hot encoding 했을 때"""
    df['Cold Forging microstructure'] = df['Cold Forging microstructure'].fillna(0)
    df.loc[(df['Cold Forging microstructure'] == 'OK', 'Cold Forging microstructure')] = 1

    # Cold Forging grain size 전처리
    """Cold Forging grain size 을 One-Hot encoding 했을 때"""
    df['Cold Forging grain size'] = df['Cold Forging grain size'].fillna(0)
    df.loc[(df['Cold Forging grain size'] == 7, 'Cold Forging grain size')] = 1


    # Normalizing temperature 전처리
    """결측치를 0 으로 채웠을 때"""
    df['Normalizing temperature'] = df['Normalizing temperature'].fillna(0)
    """결측치를 870 으로 채웠을 때"""
    # df['Normalizing temperature'] = df['Normalizing temperature'].fillna(870)

    # Normalizing time 전처리
    """결측치를 0 으로 채웠을 때"""
    df['Normalizing time'] = df['Normalizing time'].fillna(0)
    """결측치를 870 으로 채웠을 때"""
    # df['Normalizing time'] = df['Normalizing time'].fillna(120)

    # Normalizing Hardness 전처리
    """결측치를 0 으로 채웠을 때"""
    df['Normalizing Hardness'] = df['Normalizing Hardness'].fillna(0)
    """결측치를 mean 값으로 채웠을 때"""
    # df['Normalizing Hardness'] = df['Normalizing Hardness'].fillna(df['Normalizing Hardness'].mean())
    """결측치를 최빈값 으로 채웠을 때"""
    # df['Normalizing Hardness'] = df['Normalizing Hardness'].fillna(df['Normalizing Hardness'].mode())

    # SHANK RUN OUT R1 전처리
    # 모든 값이 0이라 데이터 간 구분이 불가능 = 삭제

    # SHANK RUN OUT R3 전처리
    """결측치를 0으로 채웠을 때"""
    df['SHANK RUN OUT R3'] = df['SHANK RUN OUT R3'].fillna(0)
    """결측치를 mean 값으로 채웠을 때"""
    # df['SHANK RUN OUT R3'] = df['SHANK RUN OUT R3'].fillna(df['SHANK RUN OUT R3'].mean())
    """결측치를 최빈값으로 채웠을 때"""
    # df['SHANK RUN OUT R3'] = df['SHANK RUN OUT R3'].fillna(df['SHANK RUN OUT R3'].mode())


    # SHANK 경도(HRc) 표면 전처리
    # 이 feature 부터는 ~ 표시의 값이 존재한다.
    # 우선 ~표시가 들어가는 값들은 중앙 값으로 대체한 뒤에 결측치를 채워 넣을 것이다.
    value = 0
    def pre_HRc(x):
        global value
        if str(x).find('~') == 2 or str(x).find('~') == 1:
            c, d = map(int, str(x).split('~'))
            value = (c+d)/2
            return value
        else:
            return x
    df['SHANK 경도(HRc) 표면'] = df['SHANK 경도(HRc) 표면'].apply(pre_HRc)

    """결측치를 0으로 채웠을 때"""
    df['SHANK 경도(HRc) 표면'] = df['SHANK 경도(HRc) 표면'].fillna(0)
    """결측치를 mean 값으로 채웠을 때"""
    # df['SHANK 경도(HRc) 표면'] = df['SHANK 경도(HRc) 표면'].fillna(df['SHANK 경도(HRc) 표면'].mean())
    """결측치를 최빈값으로 채웠을 때"""
    # df['SHANK 경도(HRc) 표면'] = df['SHANK 경도(HRc) 표면'].fillna(df['SHANK 경도(HRc) 표면'].mode())

    # SHANK 경도(HRc) 심부 전처리
    # ~ 값이 존재, 중앙 값 대체 후 결측치 처리

    value = 0
    def pre_HRc(x):
        global value
        if '~' in str(x):
            c, d = map(int, str(x).split('~'))
            value = (c+d)/2
            return value
        else:
            return x
    df['SHANK 경도(HRc) 심부'] = df['SHANK 경도(HRc) 심부'].apply(pre_HRc)

    """결측치를 0으로 채웠을 때"""
    df['SHANK 경도(HRc) 심부'] = df['SHANK 경도(HRc) 심부'].fillna(0)
    """결측치를 mean 값으로 채웠을 때"""
    # df['SHANK 경도(HRc) 심부'] = df['SHANK 경도(HRc) 심부'].fillna(df['SHANK 경도(HRc) 심부'].mean())
    """결측치를 최빈값으로 채웠을 때"""
    # df['SHANK 경도(HRc) 심부'] = df['SHANK 경도(HRc) 심부'].fillna(df['SHANK 경도(HRc) 심부'].mode())


    # BOWL RUN OUT R6 전처리

    """결측치를 0으로 채웠을 때"""
    df['BOWL RUN OUT R6'] = df['BOWL RUN OUT R6'].fillna(0)
    """결측치를 mean 값으로 채웠을 때"""
    # df['BOWL RUN OUT R6'] = df['BOWL RUN OUT R6'].fillna(df['BOWL RUN OUT R6'].mean())
    """결측치를 최빈값으로 채웠을 때"""
    # df['BOWL RUN OUT R6'] = df['BOWL RUN OUT R6'].fillna(df['BOWL RUN OUT R6'].mode())

    # BOWL RUN OUT R5-R4 전처리

    """결측치를 0으로 채웠을 때"""
    df['BOWL RUN OUT R5-R4'] = df['BOWL RUN OUT R5-R4'].fillna(0)
    """결측치를 mean 값으로 채웠을 때"""
    # df['BOWL RUN OUT R5-R4'] = df['BOWL RUN OUT R5-R4'].fillna(df['BOWL RUN OUT R5-R4'].mean())
    """결측치를 최빈값으로 채웠을 때"""
    # df['BOWL RUN OUT R5-R4'] = df['BOWL RUN OUT R5-R4'].fillna(df['BOWL RUN OUT R5-R4'].mode())


    # BOWL 경도(HRc) 표면 전처리
    value = 0
    def pre_HRc(x):
        global value
        if str(x).find('~') == 2:
            c, d = map(int, str(x).split('~'))
            value = (c+d)/2
            return value
        else:
            return x
    df['BOWL 경도(HRc) 표면'] = df['BOWL 경도(HRc) 표면'].apply(pre_HRc)

    """결측치를 0으로 채웠을 때"""
    df['BOWL 경도(HRc) 표면'] = df['BOWL 경도(HRc) 표면'].fillna(0)
    """결측치를 mean 값으로 채웠을 때"""
    # df['BOWL 경도(HRc) 표면'] = df['BOWL 경도(HRc) 표면'].fillna(df['BOWL 경도(HRc) 표면'].mean())
    """결측치를 최빈값으로 채웠을 때"""
    # df['BOWL 경도(HRc) 표면'] = df['BOWL 경도(HRc) 표면'].fillna(df['BOWL 경도(HRc) 표면'].mode())

    # BOWL 경도(HRc) 심부 전처리
    # 이 데이터는 ~ 표시도 포함되어 있고 조 표시 여부도 있다.
    # 이를 완전히 처리하는 것은 어려울 것으로 보이므로 생략할 것이다.

    # cycle 타임에 outlier 존재.

    # 데이터 전처리를 어떻게 할 것인지는 rainforest와 xgboost 두 개의 모델을 통해서 비교할 것이다.

    target_list = ['G','T1','T2','W_R','W_L']
    target = ['G']

    df = df.drop(labels='Cold Forging microstructure', axis=1)

    x = df.drop(labels=target_list, axis=1)
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42 ,shuffle=True)

    return df
