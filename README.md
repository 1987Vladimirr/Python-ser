import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder

# Инициализация начальных данных
lst = ['robot'] * 10
lst += ['human'] * 10
random.shuffle(lst)
data = pd.DataFrame({'whoAmI': lst})

# Создание экземпляра OneHotEncoder
encoder = OneHotEncoder()

# Передаём DataFrame в encoder.fit_transform(), используя в качестве X столбец 'whoAmI'
# .values.reshape(-1, 1) преобразует данные в формат, приемлемый для OneHotEncoder
# Pandas >= 0.24 позволяет использовать .to_numpy() вместо .values
X = data[['whoAmI']]

# Кодирование данных
encoded_data = encoder.fit_transform(X).toarray()

# Получение названий новых столбцов на основе уникальных значений категорий
columns = encoder.get_feature_names_out(input_features=['whoAmI'])

# Преобразование one-hot encoded массива обратно в DataFrame для удобства
encoded_df = pd.DataFrame(encoded_data, columns=columns)

# Вывод результирующего DataFrame
print(encoded_df.head())
