# Звіт: EDA та Нормалізація Даних
### SaaS Churn Prediction — MVP Pipeline | 2026-06-24

---

## 1. Вхідні дані

**Файл:** `data/saas_master_dataset_x6.csv`  
**Розмір:** 3 000 рядків × 37 колонок  
**Цільова змінна:** `churn_flag` (бінарна: 0 = лояльний, 1 = відтік)

### Структура датасету

| Група ознак | Колонки |
|---|---|
| **Ідентифікатор** | `account_id` |
| **Профіль клієнта** | `industry`, `country`, `referral_source`, `seats`, `is_trial`, `tenure_days`, `is_reactivation` |
| **Фінанси** | `plan_tier`, `mrr_amount`, `arr_amount`, `billing_frequency`, `refund_amount_usd` |
| **Підписка** | `upgrade_flag`, `downgrade_flag`, `auto_renew_flag`, `subscription_days`, `plan_changes_count`, `preceding_upgrade_flag`, `preceding_downgrade_flag` |
| **Активність** | `total_usage_events`, `total_duration_secs`, `total_errors`, `unique_features_used`, `unique_days_active`, `beta_feature_usage`, `days_since_last_use` |
| **Підтримка** | `support_tickets_count`, `avg_resolution_hours`, `avg_satisfaction_score`, `escalated_tickets`, `avg_first_response_mins`, `high_priority_tickets`, `has_open_ticket`, `has_support_contact`, `reason_code` |
| **Target** | `churn_flag` |

---

## 2. Дисбаланс класів

> [!IMPORTANT]
> Датасет **незбалансований**: 22% відтоку проти 78% лояльних. При побудові моделей необхідно враховувати це через `class_weight='balanced'`, SMOTE або threshold-tuning.

| Клас | Кількість | Частка |
|------|-----------|--------|
| 0 — Лояльні | 2 339 | 77.97% |
| 1 — Відтік | 661 | 22.03% |

![Дисбаланс класів churn_flag](eda_plots/01_class_imbalance.png)

---

## 3. Пропущені значення

Перед будь-якою трансформацією зафіксовано такі gaps:

| Колонка | Пропущено | % від 3 000 | Причина |
|---------|-----------|-------------|---------|
| `reason_code` | 2 555 | 85.2% | Заповнюється лише при відтоку/зверненні |
| `refund_amount_usd` | 2 555 | 85.2% | Є лише при поверненні коштів |
| `preceding_upgrade_flag` | 2 200 | 73.3% | Лише для акаунтів з попередньою підпискою |
| `preceding_downgrade_flag` | 2 200 | 73.3% | Аналогічно |
| `arr_amount` | 466 | 15.5% | Відсутній для частини тарифів |
| `mrr_amount` | 466 | 15.5% | Пов'язане з `arr_amount` |
| `avg_satisfaction_score` | 206 | 6.9% | Клієнти без оцінок |
| `avg_first_response_mins` | 40 | 1.3% | Клієнти без тікетів |
| `avg_resolution_hours` | 40 | 1.3% | Аналогічно |

**Стратегія заповнення:** числові → **медіана** колонки; категоріальні → значення `"Unknown"`.

---

## 4. Аналіз фінансових метрик (MRR)

### MRR за групами

| Група | Count | Mean MRR | Median MRR | Max MRR |
|-------|-------|----------|------------|---------|
| Лояльні (0) | 1 953 | $3 198 | $1 433 | $73 806 |
| Відтік (1) | 581 | $3 041 | $1 764 | $30 046 |

> [!NOTE]
> Медіана MRR у групі відтоку ($1 764) **вища**, ніж у лояльних ($1 433). Це вказує на те, що відтік не концентрується у найдешевших сегментах — можлива проблема продуктового fit на Pro/Enterprise тарифах.

![Розподіл MRR у розрізі відтоку](eda_plots/02_mrr_by_churn.png)

---

## 5. Кореляційний аналіз

### Топ-10 ознак за кореляцією з `churn_flag`

| Ознака | \|r\| з churn_flag | Інтерпретація |
|--------|--------------|---------------|
| `preceding_upgrade_flag` | **0.429** | Попереднє підвищення тарифу — сильний предиктор відтоку |
| `preceding_downgrade_flag` | 0.156 | Пониження тарифу перед відтоком |
| `days_since_last_use` | 0.111 | Довга відсутність активності → ризик |
| `beta_feature_usage` | 0.090 | Використання бета → лояльніші клієнти |
| `total_duration_secs` | 0.084 | Загальний час у продукті |
| `total_usage_events` | 0.082 | Кількість дій у продукті |
| `unique_days_active` | 0.079 | Днів активного використання |
| `unique_features_used` | 0.078 | Глибина освоєння продукту |
| `high_priority_tickets` | 0.076 | Критичні тікети підтримки |
| `upgrade_flag` | 0.071 | Поточне підвищення тарифу |

![Кореляційна матриця числових ознак](eda_plots/03_correlation_heatmap.png)

> [!TIP]
> `preceding_upgrade_flag` (|r| = 0.43) — найсильніший сигнал. Клієнти, які нещодавно перейшли на вищий тариф і потім відтекли, — окремий сегмент для глибшого аналізу (можливо, неправильний онбординг на новий тариф).

---

## 6. Аналіз підтримки vs Задоволеність

![Підтримка vs Задоволеність за статусом клієнта](eda_plots/04_support_vs_satisfaction.png)

**Висновки зі scatter plot:**
- Клієнти з **великою кількістю тікетів та низькою оцінкою задоволеності** концентруються у групі відтоку.
- Клієнти **без тікетів або з оцінкою > 4.0** переважно лояльні.
- Видима горизонтальна смуга при `support_tickets_count = 0` — клієнти, що ніколи не зверталися до підтримки, здебільшого залишаються.

---

## 7. Text / Categorical Feature Engineering

**Ситуація:** Колонка `last_support_text` (вільний текст тікету) відсутня у цьому датасеті.  
**Рішення:** Витягнуто два proxy-ознаки:

| Нова ознака | Джерело | Логіка |
|-------------|---------|--------|
| `text_length` | `reason_code` (довжина рядка) | Проксі наявності текстового пояснення відтоку |
| `has_ticket` | `support_tickets_count > 0` | Бінарний флаг наявності будь-якого тікету |

- `has_ticket = 1` → **2 960 з 3 000 клієнтів** мали хоча б один тікет (98.7%)
- `text_length`: медіана = 0 (більшість без `reason_code`), max = 10 символів

---

## 8. Категоріальні ознаки

| Колонка | Унікальних | Топ-3 значення |
|---------|------------|----------------|
| `industry` | 5 | DevTools (712), FinTech (687), HealthTech (586) |
| `country` | 7 | US (1797), UK (331), IN (280) |
| `referral_source` | 5 | organic (681), ads (614), other (605) |
| `plan_tier` | 3 | Enterprise (1040), Basic (985), Pro (975) |
| `billing_frequency` | 2 | annual (1635), monthly (1365) |

**Обробка:** `pd.get_dummies(..., drop_first=True)` → 5 категоріальних колонок розгорнуто в **17 бінарних ознак** (уникнення мультиколінеарності).

---

## 9. Нормалізація числових ознак

**Метод:** `StandardScaler` (z-score normalization): `x_scaled = (x - mean) / std`

- Застосовано до **31 числової колонки**
- Після масштабування: `mrr_amount` → mean ≈ 0.0000, std ≈ 1.0002 ✓
- **Мета:** підготовка для LogisticRegression Baseline (CatBoost масштабування не потребує, але єдиний препроцесований датасет зручніший для порівняння моделей)

---

## 10. Фінальний датасет

**Файл:** `data/model_ready_data.csv`

| Параметр | До обробки | Після обробки |
|----------|-----------|---------------|
| Рядків | 3 000 | 3 000 |
| Колонок | 37 | **49** |
| Пропущених значень | 10 327 | **0** |
| Числових ознак | 29 | 31 (+`text_length`, `has_ticket`) |
| Категоріальних ознак | 5 | 0 (→ 17 бінарних OHE) |

### Нові колонки після OHE (приклад)

```
industry_DevTools, industry_FinTech, industry_HealthTech, industry_HRTech,
country_IN, country_UK, country_US, ...,
plan_tier_Enterprise, plan_tier_Pro,
billing_frequency_monthly
```

---

## 11. Висновки та наступні кроки

> [!IMPORTANT]
> Ключові бізнес-інсайти з EDA:
> 1. **preceding_upgrade_flag** — найсильніший предиктор відтоку (r=0.43). Клієнти після апгрейду потребують посиленого онбордингу.
> 2. **days_since_last_use** — чіткий сигнал: неактивні клієнти відтікають частіше.
> 3. MRR сам по собі — слабкий предиктор; важливіша **поведінкова активність**.
> 4. Дисбаланс 78/22 — обов'язково використовувати `class_weight='balanced'` або SMOTE.

**Наступний крок:** Побудова моделей на `model_ready_data.csv`:
- **Baseline:** LogisticRegression
- **Основні моделі:** CatBoost / XGBoost
- **Інтерпретація:** SHAP values для топ-ознак
