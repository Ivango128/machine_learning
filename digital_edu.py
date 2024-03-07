#создай здесь свой индивидуальный проект!
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
df.drop(['has_mobile','id','has_photo','langs','last_seen','career_start','career_end'], axis = 1, inplace = True)
#print(df.head(30))
#df['sex'].value_counts().head(10)
#df.pivot_table(columns = 'sex', values = 'result').head(10).plot(kind = 'hist')

#plt.show()
a = df['bdate'].value_counts()


def fill_age(row):
    if len(str(row['bdate'])) >= 5:
        try:
            if int(str(row['bdate'])[len(str(row['bdate']))-4:len(str(row['bdate']))]) > 1968:
                #print(str(row['bdate'])[len(str(row['bdate']))-4:len(str(row['bdate']))])
                return 2024 - int(str(row['bdate'])[len(str(row['bdate']))-4:len(str(row['bdate']))])
        except:
            return 35
            #print(str(row['bdate'])[len(str(row['bdate']))-4:len(str(row['bdate']))])
    return 35



df['age'] = df.apply(fill_age, axis = 1)
df.drop(['bdate'], axis =1, inplace= True)

# age_by = df[df['result']==1]['age'].value_counts()

# age_by.drop(0, inplace = True)
# age_by = age_by.sort_index()

# age_by.plot(kind = 'bar')

# plt.show()



df = df[df.age > 0]

#type_by = df[df['result']==1]['occupation_type'].value_counts()

#type_by.plot(kind = 'barh')

#plt.show()

df['followers_count'] = df['followers_count'].apply(int)
#
# followers_count_by = df[df['result']==1]['followers_count'].value_counts()
#
# followers_count_by_transposed = followers_count_by.sort_index()
#
# plt.bar(followers_count_by_transposed.index, followers_count_by_transposed.values)
# plt.xscale('log')
# plt.show()

df['relation'] = df['relation'].apply(int)
df['graduation'] = df['graduation'].apply(int)
# relation_count_by = df[df['result']==1]['relation'].value_counts()
# print(relation_count_by)
#
# plt.bar(relation_count_by.index, relation_count_by.values)
# plt.show()


# relation_count_by = df[df['result']==1]['education_status'].value_counts()
# plt.barh(relation_count_by.index, relation_count_by.values)
# #plt.xticks(rotation=45)
# plt.show()

#print(df['langs'].value_counts())

# relation_count_by = df[df['result']==1]['langs'].value_counts()
# print(relation_count_by)
# plt.barh(relation_count_by.index, relation_count_by.values)
# plt.show()



# relation_count_by = df[df['result']==1]['city'].value_counts()
# relation_count_by = relation_count_by[relation_count_by > 9]
#
# plt.pie(relation_count_by)
#
# plt.gca().set_aspect("equal")
#
# legend_labels = [f"{city} ({count})" for city, count in zip(relation_count_by.index, relation_count_by.values)]
# plt.gca().legend(legend_labels, bbox_to_anchor=(1, 0.5), loc='center left')
#
# plt.gca().set_title('City Distribution')
#
# plt.gca().set_prop_cycle(None)
# for t in plt.gca().texts:
#     t.set_rotation(45)
#
# plt.show()


# relation_count_by = df[df['result']==1]['career_end'].value_counts()
# relation_count_by = relation_count_by[relation_count_by > 0]
#
# plt.pie(relation_count_by)
#
# plt.gca().set_aspect("equal")
#
# legend_labels = [f"{city} ({count})" for city, count in zip(relation_count_by.index, relation_count_by.values)]
# plt.gca().legend(legend_labels, bbox_to_anchor=(1, 0.5), loc='center left')
#
# plt.gca().set_title('End year')
#
# plt.gca().set_prop_cycle(None)
# for t in plt.gca().texts:
#     t.set_rotation(45)
#
# plt.show()

#print(df['education_form'].value_counts())

pd.get_dummies(df['education_form'])
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop('education_form', axis = 1, inplace = True)

df['city'] = df['city'].map({'False': 0, 'True': 1})
pd.get_dummies(df['city'])
df[list(pd.get_dummies(df['city']).columns)] = pd.get_dummies(df['city'])
df.drop('city', axis = 1, inplace = True)

df['relation'] = df['relation'].map({'False': 0, 'True': 1})
pd.get_dummies(df['relation'])
df[list(pd.get_dummies(df['relation']).columns)] = pd.get_dummies(df['relation'])
df.drop('relation', axis = 1, inplace = True)

pd.get_dummies(df['education_status'])
df[list(pd.get_dummies(df['education_status']).columns)] = pd.get_dummies(df['education_status'])
df.drop('education_status', axis = 1, inplace = True)

#df['occupation_type'] = df['occupation_type'].map({'False': 0, 'True': 1})
pd.get_dummies(df['occupation_type'])
df[list(pd.get_dummies(df['occupation_type']).columns)] = pd.get_dummies(df['occupation_type'])
df.drop('occupation_type', axis = 1, inplace = True)

#df['life_main'] = df['life_main'].map({'False': 0, 'True': 1})
pd.get_dummies(df['life_main'])
df[list(pd.get_dummies(df['life_main']).columns)] = pd.get_dummies(df['life_main'])
df.drop('life_main', axis = 1, inplace = True)

pd.get_dummies(df['people_main'])
df[list(pd.get_dummies(df['people_main']).columns)] = pd.get_dummies(df['people_main'])
df.drop('people_main', axis = 1, inplace = True)
# df['life_main'] = df['life_main'].replace('False', 0)
# df['people_main'] = df['people_main'].replace('False', 0)

df['occupation_name'] = df['occupation_name'].map({'False': 0, 'True': 1})
pd.get_dummies(df['occupation_name'])
df[list(pd.get_dummies(df['occupation_name']).columns)] = pd.get_dummies(df['occupation_name'])
df.drop('occupation_name', axis = 1, inplace = True)



# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Шаг 2. Создание модели

df.columns = df.columns.astype(str)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


X = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))

TP, TN, FP, FN = 0, 0, 0, 0

for test, pred in zip(y_test, y_pred):
    if test - pred == 0:
        if test == 1:
            TP += 1
        else:
            TN += 1
    else:
        if test == 1:
            FN += 1
        else:
            FP += 1


#алгоритм распределения по категориям
print('Верный прогноз: купившие -', TP, 'не купившие -', TN)
print('Ошибочный прогноз: купившие -', FP, 'не купившие -', FN)