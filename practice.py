import pandas as pd

df = pd.read_excel('FHK2022_dataA.xlsx')
df = df.drop(labels=['Work Time', 'Serial No.', 'Material Date',
                         'SHANK RUN OUT R1', 'BOWL 경도(HRc) 심부', 'Part Name'], axis=1)
df = df.loc[:, (df != df.iloc[0]).any()]

print(df)
values = df.values
df = pd.DataFrame(values, columns=list(range(df.shape[1])))
print(df)