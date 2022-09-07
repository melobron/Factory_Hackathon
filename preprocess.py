import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from nn import preprocess2

def data_report(process=True):
    if process:
        x, y, x1, x2, x3 = preprocess(output_param='G')
    else:
        x = pd.read_excel(filename='FHK2022_dataA.xlsx')
    report = ProfileReport(x)
    report.to_file('./report.html')


def normalization(args, df):
    if args.norm_type == 'MinMax':
        scaler = MinMaxScaler()
    elif args.norm_type == 'Standard':
        scaler = StandardScaler()
    else:
        raise NotImplementedError('Wrong type')

    output = scaler.fit_transform(df)
    df = pd.DataFrame(output, columns=df.columns, index=list(df.index.values))
    return df, scaler


def remove_outliers(args, df):
    outlier = LocalOutlierFactor(n_neighbors=args.n_neighbors, metric=args.metric, contamination=args.contamination)
    df['outlier'] = outlier.fit_predict(df)
    df_search = df[df['outlier'] == -1]
    df = df.drop(df_search.index, axis=0)
    df = df.drop(['outlier'], axis=1)
    return df


def preprocess(args, filename='FHK2022_dataA.xlsx', output_param='G'):
    df = pd.read_excel(filename)

    # Remove 2 columns
    df = df.drop(labels=['Work Time', 'Serial No.', 'Material Date',
                         'SHANK RUN OUT R1', 'BOWL 경도(HRc) 심부', 'Part Name'], axis=1)

    # # Remove columns with Nan
    # null_cols = df.columns[df.isnull().any()]
    # df = df.drop(null_cols, axis=1)

    # Remove columns with a constant value
    df = df.loc[:, (df != df.iloc[0]).any()]

    # Columns preprocess
    df['Quenching Density'] = df['Quenching Density'].fillna(0)
    df['Surface Hardness One'] = 0
    df.loc[(df['Surface Hardness'] == 7.8, 'Surface Hardness')] = 1
    df = df.drop(labels=['Surface Hardness'], axis=1)
    df['Material  Heat No.'] = df['Material  Heat No.'].fillna('non_value')
    for i in range(df['Material  Heat No.'].nunique()):
        df['Material Heat No.'+str(i)] = 0
        df.loc[(df['Material  Heat No.'] == df['Material  Heat No.'].unique()[i], 'Material Heat No.'+str(i))] = 1
    df = df.drop(labels='Material  Heat No.', axis=1)
    col_name = 'C(%, X100)'
    df[col_name] = df[col_name].fillna(0)
    col_name = 'Si(%, X100)'
    df[col_name] = df[col_name].fillna(0)
    col_name = 'Mn(%, X100)'
    df[col_name] = df[col_name].fillna(0)
    col_name = 'P(%, X100)'
    df[col_name] = df[col_name].fillna(0)
    col_name = 'S(%, X100)'
    df[col_name] = df[col_name].fillna(0)
    col_name = 'Cu(%, X100)'
    df[col_name] = df[col_name].fillna(0)
    col_name = 'Ni(%, X100)'
    df[col_name] = df[col_name].fillna(0)
    col_name = 'Cr(%, X100)'
    df[col_name] = df[col_name].fillna(0)
    col_name = 'Mo(%, X100)'
    df[col_name] = df[col_name].fillna(0)
    col_name = 'Forging temperature(˚)'
    df[col_name] = df[col_name].fillna(0)
    df['Cold Forging Hardness'] = df['Cold Forging Hardness'].fillna(0)
    df['Cold Forging microstructure'] = df['Cold Forging microstructure'].fillna(0)
    df.loc[(df['Cold Forging microstructure'] == 'OK', 'Cold Forging microstructure')] = 1
    df['Cold Forging grain size'] = df['Cold Forging grain size'].fillna(0)
    df.loc[(df['Cold Forging grain size'] == 7, 'Cold Forging grain size')] = 1
    df['Normalizing temperature'] = df['Normalizing temperature'].fillna(0)
    df['Normalizing time'] = df['Normalizing time'].fillna(0)
    df['Normalizing Hardness'] = df['Normalizing Hardness'].fillna(0)
    df['SHANK RUN OUT R3'] = df['SHANK RUN OUT R3'].fillna(0)
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
    df['SHANK 경도(HRc) 표면'] = df['SHANK 경도(HRc) 표면'].fillna(0)
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
    df['SHANK 경도(HRc) 심부'] = df['SHANK 경도(HRc) 심부'].fillna(0)
    df['BOWL RUN OUT R6'] = df['BOWL RUN OUT R6'].fillna(0)
    df['BOWL RUN OUT R5-R4'] = df['BOWL RUN OUT R5-R4'].fillna(0)
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
    df['BOWL 경도(HRc) 표면'] = df['BOWL 경도(HRc) 표면'].fillna(0)
    df = df.drop(labels='Cold Forging microstructure', axis=1)

    # Input, Output
    x = df.drop(labels=['G', 'T1', 'T2', 'W_R', 'W_L'], axis=1)
    y = df.loc(axis=1)[output_param]

    # Additional process
    values = x.values
    x = pd.DataFrame(values, columns=list(range(x.shape[1])))

    # Outliers
    if args.remove_out:
        df = remove_outliers(args, pd.concat([x, y], axis=1))
        x, y = df.iloc[:, :-1], df.iloc[:, -1]

    # # Train/Test split
    # x_train, x_valid_test, y_train, y_valid_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    # x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.5, shuffle=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)

    # Normalization
    if args.normalization:
        train_before_norm = pd.concat([x_train, y_train], axis=1)
        # valid_before_norm = pd.concat([x_valid, y_valid], axis=1)
        test_before_norm = pd.concat([x_test, y_test], axis=1)

        train_after_norm, scaler = normalization(args, train_before_norm)
        # valid_after_norm = pd.DataFrame(scaler.transform(valid_before_norm), columns=train_after_norm.columns)
        test_after_norm = pd.DataFrame(scaler.transform(test_before_norm), columns=train_after_norm.columns)

        x_train, y_train = train_after_norm.iloc[:, :-1], train_after_norm.iloc[:, -1]
        # x_valid, y_valid = valid_after_norm.iloc[:, :-1], valid_after_norm.iloc[:, -1]
        x_test, y_test = test_after_norm.iloc[:, :-1], test_after_norm.iloc[:, -1]

        # return [x_train, x_valid, x_test], [y_train, y_valid, y_test], scaler
        return [x_train, x_test], [y_train, y_test], scaler
    else:
        # return [x_train, x_valid, x_test], [y_train, y_valid, y_test]
        return [x_train, x_test], [y_train, y_test]
