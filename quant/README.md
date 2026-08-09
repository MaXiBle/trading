# MOEX Research / Backtest Platform

Единая research/backtest-платформа для Мосбиржи, поверх которой последовательно строятся и проверяются 6 модулей генерации и улучшения сигнала.

## Структура проекта

```
quant/
├── configs/              # Конфигурационные файлы YAML
├── src/                  # Исходный код платформы
│   ├── data_ingestion/   # Загрузка данных MOEX ISS
│   ├── data_quality/     # Проверки качества данных
│   ├── corporate_actions/# Корпоративные события и.adjustment
│   ├── features/         # Feature engineering
│   ├── labels/           # Построение labels/targets
│   ├── splitting/        # Walk-forward splitter
│   ├── backtest/         # Backtest engine
│   ├── metrics/          # Метрики и отчеты
│   ├── models/           # ML модели
│   ├── portfolio/        # Портфельные ограничения
│   └── utils/            # Утилиты
├── pipelines/            # ETL пайплайны
├── tests/                # Тесты
├── reports/              # Отчеты
├── data/                 # Данные (не версионируются)
│   ├── raw/              # Сырые данные MOEX
│   ├── curated/          # Очищенные данные
│   ├── features/         # Признаки
│   ├── labels/           # Labels
│   ├── predictions/      # Предсказания
│   └── backtests/        # Результаты бэктестов
├── Makefile
├── pyproject.toml
└── README.md
```

## Быстрый старт

### Установка зависимостей

```bash
make install
```

### Загрузка данных MOEX

```bash
make ingest_moex
```

### Построение curated данных

```bash
make build_curated
```

### Проверка качества данных

```bash
make data_quality
```

### Запуск dummy backtest

```bash
make run_dummy_backtest
```

### Запуск тестов

```bash
make test
```

## Архитектурный принцип

```
Данные → Feature store → Labels → Splitter → Backtester → Метрики
                                                      ↓
                                              baseline → модули
```

Каждый модуль обязан ответить на вопрос: **Что он добавляет к baseline после издержек?**

## Фазы разработки

| Фаза | Суть | Выходной критерий |
|------|------|-------------------|
| **0** | Платформа: данные, календарь, CA, splitter, backtester, метрики | Dummy backtest воспроизводится одной командой |
| **1** | Baseline: cross-sectional LightGBM, long-only, H=5 | Понятный baseline с cost sensitivity |
| **2** | Meta-labeling + Regime overlay | Улучшение Sharpe/maxDD или отказ от модуля |
| **3** | Event-driven news pipeline | Event dataset, event study, incremental signal |
| **4** | Options / Derivatives | Feasibility → Plan A или Plan B → incremental evaluation |
| **5** | Graph lead-lag | Graph features как добавка к baseline |

## Обязательные принципы

Запрещено:
1. Использовать будущие данные в features / labels / universe
2. Обучать или тестировать модели без embargo
3. Использовать unadjusted returns без явной пометки
4. Игнорировать издержки
5. Строить universe только по текущему составу индекса
6. Считать backtest без cost sensitivity

При любом споре выбирается более консервативный вариант без lookahead.

## Лицензия

MIT
