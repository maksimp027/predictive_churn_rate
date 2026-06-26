# describe.md — `model_ready_data.csv`

> Препроцесований датасет для задачі **бінарної класифікації відтоку SaaS-клієнтів**.  
> Сформований скриптом [`eda_pipeline.py`](./eda_pipeline.py) з вихідного файлу `data/saas_master_dataset_x6.csv`.

---

## 1. Загальна характеристика

| Параметр | Значення |
|---|---|
| Файл | `model_ready_data.csv` |
| Рядків | **3 000** |
| Колонок | **49** |
| Пропущених значень | **0** |
| Цільова змінна | `churn_flag` |
| Баланс класів | 77.97 % лояльні (0) / 22.03 % відтік (1) |
| Тип задачі | Бінарна класифікація |

---

## 2. Конвеєр препроцесингу

```
saas_master_dataset_x6.csv
    │
    ├── Видалення технічних колонок: account_id
    ├── Text Feature Engineering:
    │       text_length  ← len(reason_code)
    │       has_ticket   ← support_tickets_count > 0
    ├── Заповнення пропущених числових значень медіаною
    ├── StandardScaler  → усі числові ознаки (mean≈0, std≈1)
    ├── OneHotEncoding  → категоріальні (drop_first=True, dtype=int)
    │       industry, country, referral_source, plan_tier, billing_frequency
    └── model_ready_data.csv
```

> **Примітка:** Числові ознаки стандартизовані (`sklearn.preprocessing.StandardScaler`), тому їхні значення є Z-оцінками, а не оригінальними одиницями.

---

## 3. Словник ознак

### 3.1 Ознаки підписки (Subscription Features)

Стандартизовані числові значення (Z-score).

| Колонка | Тип | Опис |
|---|---|---|
| `seats` | float (scaled) | Кількість ліцензованих місць (користувачів). Оригінальний діапазон: 1–120+ |
| `is_trial` | float (scaled) | Чи перебуває клієнт на пробному тарифі (0/1 до масштабування) |
| `tenure_days` | float (scaled) | Загальний термін перебування клієнта (дні) |
| `mrr_amount` | float (scaled) | Monthly Recurring Revenue — місячний регулярний дохід (USD) |
| `arr_amount` | float (scaled) | Annual Recurring Revenue — річний регулярний дохід (USD). `arr = mrr × 12` |
| `upgrade_flag` | float (scaled) | Чи здійснював клієнт апгрейд тарифу (0/1) |
| `downgrade_flag` | float (scaled) | Чи здійснював клієнт даунгрейд тарифу (0/1) |
| `auto_renew_flag` | float (scaled) | Чи увімкнено автоматичне поновлення підписки (0/1) |
| `subscription_days` | float (scaled) | Тривалість поточного підписного циклу (дні) |
| `plan_changes_count` | float (scaled) | Кількість змін тарифного плану за весь час |
| `preceding_upgrade_flag` | float (scaled) | Чи був апгрейд у попередньому циклі підписки (0/1) |
| `preceding_downgrade_flag` | float (scaled) | Чи був даунгрейд у попередньому циклі підписки (0/1) |
| `is_reactivation` | float (scaled) | Клієнт реактивований після попереднього відтоку (0/1) |
| `refund_amount_usd` | float (scaled) | Сума повернень коштів клієнту (USD) |

---

### 3.2 Ознаки використання продукту (Product Usage Features)

| Колонка | Тип | Опис |
|---|---|---|
| `total_usage_events` | float (scaled) | Загальна кількість подій взаємодії з продуктом |
| `total_duration_secs` | float (scaled) | Загальний час використання (секунди) |
| `total_errors` | float (scaled) | Кількість зафіксованих помилок під час роботи |
| `unique_features_used` | float (scaled) | Кількість унікальних функцій/модулів, що використовувались |
| `unique_days_active` | float (scaled) | Кількість унікальних днів активності |
| `beta_feature_usage` | float (scaled) | Кількість використань бета-функцій |
| `days_since_last_use` | float (scaled) | Кількість днів з моменту останньої активності |

---

### 3.3 Ознаки підтримки (Support Features)

| Колонка | Тип | Опис |
|---|---|---|
| `support_tickets_count` | float (scaled) | Загальна кількість тікетів у службу підтримки |
| `avg_resolution_hours` | float (scaled) | Середній час вирішення тікету (години) |
| `avg_satisfaction_score` | float (scaled) | Середня оцінка задоволеності підтримкою (шкала 1–5) |
| `escalated_tickets` | float (scaled) | Кількість ескальованих тікетів |
| `avg_first_response_mins` | float (scaled) | Середній час першої відповіді (хвилини) |
| `high_priority_tickets` | float (scaled) | Кількість тікетів із високим пріоритетом |
| `has_open_ticket` | float (scaled) | Чи є наразі відкритий тікет (0/1). У датасеті всі значення = 0 |
| `has_support_contact` | float (scaled) | Клієнт мав хоча б один контакт із підтримкою (0/1) |

---

### 3.4 Текстові / NLP-ознаки (Text-derived Features)

Отримані з колонки `reason_code` вихідного датасету.

| Колонка | Тип | Опис |
|---|---|---|
| `text_length` | float (scaled) | Довжина рядка `reason_code` (символів). Проксі для деталізованості скарги/відповіді. 5 унікальних значень після масштабування |
| `has_ticket` | int (0/1) | Клієнт має хоча б 1 тікет підтримки (`support_tickets_count > 0`). 2960 з 3000 = 1 |

---

### 3.5 OHE-категоріальні ознаки (One-Hot Encoded)

Бінарні індикатори (0/1, `dtype=int`). Базова категорія кожної групи видалена (`drop_first=True`).

#### Індустрія клієнта (`industry_*`)

Базова категорія — `industry_Consulting` (або перша алфавітно).

| Колонка | Розподіл (1/0) | Опис |
|---|---|---|
| `industry_DevTools` | 712 / 2 288 | Клієнт з індустрії DevTools |
| `industry_EdTech` | 436 / 2 564 | Клієнт з індустрії EdTech |
| `industry_FinTech` | 687 / 2 313 | Клієнт з індустрії FinTech |
| `industry_HealthTech` | 586 / 2 414 | Клієнт з індустрії HealthTech |

#### Країна клієнта (`country_*`)

Базова категорія — `country_AU` (або перша алфавітно).

| Колонка | Розподіл (1/0) | Опис |
|---|---|---|
| `country_CA` | 141 / 2 859 | Канада |
| `country_DE` | 133 / 2 867 | Німеччина |
| `country_FR` | 130 / 2 870 | Франція |
| `country_IN` | 280 / 2 720 | Індія |
| `country_UK` | 331 / 2 669 | Велика Британія |
| `country_US` | 1 797 / 1 203 | США (найбільша частка) |

#### Джерело залучення (`referral_source_*`)

Базова категорія — `referral_source_ads` (або перша алфавітно).

| Колонка | Розподіл (1/0) | Опис |
|---|---|---|
| `referral_source_event` | 558 / 2 442 | Залучений через захід/конференцію |
| `referral_source_organic` | 681 / 2 319 | Органічний пошук |
| `referral_source_other` | 605 / 2 395 | Інше джерело |
| `referral_source_partner` | 542 / 2 458 | Партнерська програма |

#### Тарифний план (`plan_tier_*`)

Базова категорія — `plan_tier_Basic`.

| Колонка | Розподіл (1/0) | Опис |
|---|---|---|
| `plan_tier_Enterprise` | 1 040 / 1 960 | Enterprise тариф |
| `plan_tier_Pro` | 975 / 2 025 | Pro тариф |

#### Частота білінгу (`billing_frequency_*`)

Базова категорія — `billing_frequency_annual`.

| Колонка | Розподіл (1/0) | Опис |
|---|---|---|
| `billing_frequency_monthly` | 1 365 / 1 635 | Щомісячний білінг |

---

### 3.6 Цільова змінна

| Колонка | Тип | Опис |
|---|---|---|
| `churn_flag` | int (0/1) | **0** — клієнт лояльний; **1** — клієнт залишив сервіс (відтік) |

| Клас | Кількість | Частка |
|---|---|---|
| 0 (лояльний) | 2 339 | 77.97 % |
| 1 (відтік) | 661 | 22.03 % |

> ⚠️ **Дисбаланс класів:** ~3.5:1. При навчанні моделей рекомендується використовувати `class_weight='balanced'`, SMOTE, або відповідну метрику (ROC-AUC, F1, PR-AUC).

---

## 4. Якість даних

| Перевірка | Результат |
|---|---|
| Пропущені значення | ✅ 0 (заповнені медіаною до масштабування) |
| Дублікати рядків | ✅ Не виявлено |
| Масштабування числових ознак | ✅ StandardScaler (mean≈0, std≈1) |
| Кодування категоріальних | ✅ OneHotEncoding, drop_first=True |
| Витік даних (leakage) | ⚠️ `arr_amount = mrr_amount × 12` — пряма дублікація; розглянути видалення однієї з двох при навчанні |
| `has_open_ticket` | ⚠️ Нульова дисперсія — всі значення = 0. Ознака не несе інформації; рекомендується видалити |

---

## 5. Рекомендовані наступні кроки

1. **Видалити** `arr_amount` або `mrr_amount` (ідеально корелюють).
2. **Видалити** `has_open_ticket` (нульова дисперсія).
3. **Розбити** на train/test (наприклад, 80/20, `stratify=churn_flag`).
4. **Обрати метрику:** ROC-AUC або PR-AUC з огляду на дисбаланс класів.
5. **Baseline модель:** `LogisticRegression(class_weight='balanced')` — дані вже стандартизовані.
6. **Просунуті моделі:** `XGBoostClassifier`, `LightGBM`, `RandomForest` (для tree-based моделей StandardScaler не є обов'язковим, але не шкодить).

---

*Сформовано автоматично на основі аналізу `model_ready_data.csv` та `eda_pipeline.py`.*
