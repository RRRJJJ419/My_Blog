---
title: Titanic - Machine Learning from Disaster
---
---
創建一個機器學習模型，預測在鐵達尼號沉船事故中的存活率.

Dataset Description : [Titanic](https://www.kaggle.com/competitions/titanic/data)

<br>

#### 套件安裝:
```bash
#pip install sklearn
import pandas as pd
import numpy as np

# 繪圖相關套件
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.gridspec as gridspec
import seaborn as sns
plt.style.use('ggplot')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # 標籤編碼(Label)、獨熱編碼(OneHot)
from sklearn.tree import DecisionTreeClassifier # 決策樹(Decision Tree)
from sklearn.ensemble import RandomForestClassifier # 隨機森林(Random Forest)

from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
```

<br>

#### 讀取檔案:
``` bash
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
submit = pd.read_csv('gender_submission.csv')
```
Training Data:
<img src="/My_Blog/images/Titanic_train_data.png" alt="圖片說明"/>
Testing Data:
<img src="/My_Blog/images/Titanic_test_data.png" alt="圖片說明"/>
Submit:
<img src="/My_Blog/images/Titanic_submit.png" alt="圖片說明" width="150" height="150"/>

<br>

---

- ### 資料探索(EDA)=====
#### 資料型態: 
``` bash
display(df_train.dtypes) # 使用.dtypes查看欄位資料型態
display(df_test.dtypes)
```

<br>

#### 資料缺失值數量: 
```bash
display(df_train.isnull().sum()) # 使用.isnull()判別是否為空再.sum()加總
display(df_test.isnull().sum())
```
#### 計算缺失值數量&比例
```bash
missing_train = df_train.isnull().sum()
missing_train = missing_train[missing_train > 0]
missing_train_percent = (missing_train / len(df_train)) * 100
missing_train_table = pd.DataFrame({'Missing Count': missing_train, 'Percentage (%)': missing_train_percent.round(2)})
display(missing_train_table)
```
<img src="/My_Blog/images/Titanic_data_lost.png" alt="圖片說明" width="500" height="500"/>

<br>
<br>

#### 合併訓練和測試集
```bash
df_data = pd.concat([df_train, df_test], ignore_index=True)
df_data
```

<br>

#### 欄位內容數量計算
```bash
# 存活和未存活的數量
Survived_Counts = df_data['Survived'].value_counts().reset_index()
Survived_Counts.columns = ['Survived','Counts']
Survived_Counts
```
<img src="/My_Blog/images/Titanic_survived.png" alt="圖片說明" width="150" height="150"/>

#### (圓餅圖)
```bash
plt.figure(figsize=(10,5))
df_data['Survived'].value_counts().plot(kind='pie', colors=['lightcoral','skyblue'], autopct='%1.2f%%')
plt.title('Survival')  # 圖標題
plt.ylabel('')
plt.show()
```
<img src="/My_Blog/images/Titanic_survived_piechart.png" alt="圖片說明" width="200" height="200"/>

<br>
<br>
<br>

#### 欄位間的相關係數
```bash
# 其他數值欄位和存活欄位之間的相關係數
numeric_df_train = df_train.select_dtypes(include=np.number) # 從df_train中選取出所有資料型態為數值的欄位，並建立一個新的DataFrame
Corr_Matrix = numeric_df_train.corr()  # 計算相關係數
Corr_Matrix
Corr = Corr_Matrix.loc['Survived',:].sort_values() #從Corr_Matrix中選取出'Survived'與其他所有欄位之間的相關係數，並將這些相關係數從小到大排序。[:-1]則會移除最後一個值，因為'Survived'欄位與自身的相關係數永遠是1，通常在相關性分析中會忽略這個值。
Corr = pd.DataFrame({'Survived':Corr}) #將排序後的相關係數Series轉換成一個新的DataFrame Corr，並將欄位名稱設為'Survived'。
Corr
```
<img src="/My_Blog/images/Titanic_survived_cor.png" alt="圖片說明" width="150" height="150"/>

<br>
<br>

---

- ### 特徵工程=====
#### 字串分割
```bash
df_data['Title'] = df_data.Name.str.split(', ', expand=True)[1]  # 使用(, )作為分隔符號，[1]選取分割後DataFrame的第二個欄位。根據姓名的常見格式 "LastName, Title.FirstName"，第二個部分通常包含稱謂和名字 ex:Kelly, Mr.James
df_data['Title'] = df_data.Title.str.split('.', expand=True)[0]  # 在使用(.)作為分割符號，[0]選取分割後DataFrame的第一個欄位。根據"Title.FirstName"的格式，第一個部分就是稱謂本身
df_data['Title'].unique()
```

<br>

#### 欄位類別整合取代
```bash
#將某些稱謂取代集合成另一種稱謂
df_data['Title'] = df_data.Title.replace(['Don','Rev','Dr','Major','Lady','Sir','Col','Capt','Countess','Jonkheer','Dona'], 'Rare')
df_data['Title'] = df_data.Title.replace(['Ms','Mlle'], 'Miss')
df_data['Title'] = df_data.Title.replace('Mme', 'Mrs')
df_data['Title'].unique()
```

<br>

#### 去除欄位
```bash
df_data.drop('Ticket',axis=1,inplace=True)
df_data.drop('Survived', axis=1)
```

<br>

#### 根據缺失值產生二元欄位
``` bash
#根據 'Age' 欄位是否有缺失值，建立一個新的二元欄位 'isAge'
df_data['isAge'] = df_data['Age'].isnull().map(lambda x:0 if x==True else 1)
```

<br>

#### 欄位交叉表
```bash
#計算並顯示有無年齡 'isAge' 和性別 'Sex' 欄位間的交叉表
display(pd.crosstab(df_data.isAge, df_data.Sex, margins=True))
display(pd.crosstab(df_data.isAge, df_data.Pclass, margins=True))
```
<img src="/My_Blog/images/Titanic_crosstab.png" alt="圖片說明" width="150" height="150"/>

#### (長條圖)
```bash
#繪製兩個長條圖，分別顯示年齡資訊是否缺失 ('isAge') 與性別 ('Sex') 以及年齡資訊是否缺失 ('isAge') 與艙等 ('Pclass') 之間的關係
fig, axs = plt.subplots(1,2,figsize=(14,5))

plt.subplot(1,2,1)
sns.countplot(data=df_data, x=df_data.Sex, hue=df_data.isAge, palette=['lightcoral','skyblue'])
plt.ylabel('Counts')

plt.subplot(1,2,2)
sns.countplot( data=df_data, x=df_data.Pclass, hue=df_data.isAge, palette=['lightcoral','skyblue'])
plt.ylabel('')

plt.show()
```
<img src="/My_Blog/images/Titanic_crosstab_countplot.png" alt="圖片說明" width="600" height="600"/>

<br>
<br>

#### 將一些非數值型的欄位轉換為數值型
```bash
for col in ['Title','Ticket_info','Cabin']:
    df_data[col] = df_data[col].astype('category').cat.codes # 將欄位轉換為類別型資料，.cat.codes將類別型資料轉換為數值編碼。每個類別會被賦予一個唯一的整數

df_data = pd.get_dummies(df_data, columns=['Sex', 'Embarked', 'Title', 'Ticket_info'], drop_first=True) #對指定的欄位進行One-Hot Encoding, drop_first=True會刪除每個類別的第一個One-Hot Encoding欄位，以避免多重共線性

df_data.head()
```

<br>
<br>

---

- ### 模型訓練=====
#### 訓練集/測試集準備
```bash
# 將合併資料重新分割回訓練集和測試集
Train = df_data[pd.notnull(df_data.Survived)]
Test = df_data[pd.isnull(df_data.Survived)]

Train.drop(['PassengerId', 'Name'], axis=1, inplace=True) 
Test.drop(['PassengerId','Survived', 'Name'], axis=1, inplace=True) 

Y_Train = Train.Survived  # 取將訓練集中的 Survived 單獨拆出，作為標籤Y
X_Train = Train.drop(['Survived'], axis=1)  # 刪除 'Survived' 欄位
```

<br>

#### 模型訓練
```bash
from sklearn.tree import DecisionTreeClassifier

# 初始化決策樹分類器
dt_model = DecisionTreeClassifier(random_state=42)

# 使用訓練資料訓練模型
dt_model.fit(X_Train, Y_Train)

print("決策樹模型訓練完成！")
```
#### 模型預測
```bash
# 使用訓練好的決策樹模型對測試資料進行預測
predictions = dt_model.predict(Test)

# 將預測結果轉換為整數 (0或1)
predictions = predictions.astype(int)

# 創建提交檔案的 DataFrame
submission = pd.DataFrame({
    "PassengerId": df_test['PassengerId'],
    "Survived": predictions
})
display(submission)
```
<img src="/My_Blog/images/Titanic_predict.png" alt="圖片說明" width="150" height="150"/>

<br>
<br>

#### 準確率 (Accuracy)
```bash
from sklearn.metrics import accuracy_score

# 建立一個包含 PassengerId 和預測結果的DataFrame
predictions_aligned = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': predictions})

# 合併 submit(正確答案) 和預測結果兩個DataFrame
comparison_df = pd.merge(submit, predictions_aligned, on='PassengerId', suffixes=('_true', '_pred')) #當兩個 DataFrame 中存在同名的欄位時(這裡都是'Survived')，會為它們加上後綴以區分

# 計算準確率
accuracy = accuracy_score(comparison_df['Survived_true'], comparison_df['Survived_pred'])
print(f"預測準確率為: {accuracy:.4f}")
```
