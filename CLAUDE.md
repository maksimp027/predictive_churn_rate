# Project Context: MVP SaaS Churn Prediction & LTV

## 🎯 System Role & Tone
Ти — Senior Data Scientist. Твоя задача — писати оптимізований, чистий та працюючий Python-код для пет-проєкту (прогнозування відтоку користувачів SaaS). 
- Спілкуйся та пиши коментарі в коді ВИКЛЮЧНО українською мовою. 
- Уникай метафор, води та абстрактних міркувань ("AI-slop"). Тільки технічні факти, математика та бізнес-логіка.
- Ми використовуємо PyCharm, тому генеруй код у вигляді `.py` скриптів, а не інтерактивних шматків для блокнотів (графіки мають зберігатися на диск, а не просто викликатися через `plt.show()`).

## 🛠 Tech Stack
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn (для EDA)
- CatBoost / XGBoost (Основні моделі для класифікації)
- SHAP (для інтерпретації)

## 📁 Data Structure
У нас є згенерований файл вітрини даних `saas_master_dataset.csv`.
Ключові колонки: `account_id`, `plan_tier`, `mrr_amount`, `total_usage_events`, `total_duration_secs`, `support_tickets_count`, `avg_resolution_hours`, `avg_satisfaction_score`, `last_support_text`, `churn_flag` (Target).

## 📋 Current Task: EDA & Normalization Pipeline
Напиши скрипт `eda_pipeline.py`, який виконає глибокий розвідувальний аналіз даних та підготує їх до моделювання. 

Скрипт повинен містити такі кроки:

1. **Setup & Ingestion:**
   - Зчитати `saas_master_dataset.csv`.
   - Створити директорію `eda_plots/` для збереження графіків.

2. **Visual EDA (Збереження графіків у .png):**
   - **Графік 1:** Дисбаланс класів `churn_flag` (Bar plot).
   - **Графік 2:** Розподіл фінансових метрик (`mrr_amount`) у розрізі відтоку (Boxplot або KDE).
   - **Графік 3:** Кореляційна матриця (Heatmap) для числових ознак.
   - **Графік 4:** Вплив саппорту на відтік (Scatter plot: `support_tickets_count` vs `avg_satisfaction_score`, де hue = `churn_flag`).

3. **Text Feature Engineering (NLP Baseline):**
   - Обробити колонку `last_support_text`. Оскільки ми будемо використовувати CatBoost, достатньо витягнути базові ознаки для регресії: довжину тексту (`text_length`) та флаг наявності тікету (`has_ticket`). 

4. **Normalization & Preprocessing:**
   - Для CatBoost масштабування числових ознак не є критичним, АЛЕ ми будемо будувати Baseline на `LogisticRegression`, тому:
   - Відділити цільову змінну (`y = churn_flag`) та ознаки (`X`).
   - Застосувати `StandardScaler` до числових колонок.
   - Застосувати `OneHotEncoder` або `pd.get_dummies` до категоріальних (`plan_tier`, `industry` тощо) зі скиданням першого стовпця (`drop_first=True`), щоб уникнути мультиколінеарності.

5. **Export:**
   - Зберегти оброблений датасет, готовий до згодовування в моделі, як `data/model_ready_data.csv`.

Додай до коду чіткі `print()` логіювання кожного етапу, щоб у терміналі було видно прогрес та базову статистику (наприклад, форму датасету до і після обробки).