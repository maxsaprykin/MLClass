import numpy as np
def get_bootstrap_samples(data, n_samples):
    # функция для генерации подвыборок с помощью бутстрэпа
    np.random.seed(17)
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples
def stat_intervals(stat, alpha):
    # функция для интервальной оценки
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

# сохранение в отдельные numpy массивы данных по лояльным и уже бывшим клиентам
good_income = data[data['SeriousDlqin2yrs'] == False]['MonthlyIncome'].values
bad_income= data[data['SeriousDlqin2yrs'] == True]['MonthlyIncome'].values

# ставим seed для воспроизводимости результатов
#np.random.seed(17)

# генерируем выборки с помощью бутстрэра и сразу считаем по каждой из них среднее
good_income_scores = [np.mean(sample) 
                       for sample in get_bootstrap_samples(good_income, 1000)]
bad_income_scores = [np.mean(sample) 
                       for sample in get_bootstrap_samples(bad_income, 1000)]

#  выводим интервальную оценку среднего
good_interval = stat_intervals(good_income_scores, 0.10)
bad_interval = stat_intervals(bad_income_scores, 0.10)
print("good_income:  mean interval",  good_interval)
print("bad_income:  mean interval",  bad_interval)

print("result: ", good_interval[0] - bad_interval[1])