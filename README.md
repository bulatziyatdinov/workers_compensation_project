# Проект: Прогнозирование стоимости страховых выплат
## Описание проекта
Цель проекта — разработать модель машинного обучения для предсказания
итоговой стоимости страхового возмещения (UltimateIncurredClaimCost)
на основе характеристик работника и параметров случая.
## Датасет
Используется датасет **Workers Compensation** (ID: 42876), содержащий
100,000 записей о страховых случаях.
## Установка и запуск
1. Клонируйте репозиторий:
 ```bash
git clone https://github.com/bulatziyatdinov/workers_compensation_project
```

2. Перейдите в папку проекта:
 ```bash
    cd workers_compensation_project
 ```

3. Создайте виртуальную среду:
 ```bash
    python -m venv venv
 ```

4. Активируйте виртуальную среду:
 ```bash
    ./venv/Scripts/activate
 ```

5. Установите зависимости:
 ```bash
    pip install -r requirements.txt
 ```

6. Запустите приложение:
 ```bash
    streamlit run app.py --server.port 10000
 ```

7. Перейдите по ссылке и работайте с проектом:
 ```http://localhost:10000```