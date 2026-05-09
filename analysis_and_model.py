import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@st.cache_data
def load_data() -> pd.DataFrame:
    return fetch_openml(data_id=42876, as_frame=True, parser='auto').frame


def preprocess_data(df):
    data = df.copy()
    data['DateTimeOfAccident'] = pd.to_datetime(data['DateTimeOfAccident'])
    data['DateReported'] = pd.to_datetime(data['DateReported'])
    data['AccidentMonth'] = data['DateTimeOfAccident'].dt.month
    data['AccidentDayOfWeek'] = data['DateTimeOfAccident'].dt.dayofweek
    data['ReportingDelay'] = (data['DateReported'] - data['DateTimeOfAccident']).dt.days
    data.drop(columns=['DateTimeOfAccident', 'DateReported'], inplace=True)

    cat_cols = ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'ClaimDescription']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data.drop(columns=['UltimateIncurredClaimCost'])
    y = data['UltimateIncurredClaimCost']
    feature_names = X.columns.tolist()

    return X, y, encoders, feature_names


def scale_features(X_train, X_test, numerical_cols):
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return scaler, X_train, X_test


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Реальные значения')
    ax.set_ylabel('Предсказанные значения')
    ax.set_title(f'{model_name}: Предсказания vs Реальные значения')
    plt.tight_layout()

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}, fig


def get_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return None
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    return imp_df


def main():
    st.title("Прогнозирование стоимости страховых выплат")

    if st.button("Загрузить данные"):
        with st.spinner("Загрузка данных (может занять некоторое время)..."):
            df = load_data()
            st.session_state['df'] = df
            st.success("Данные успешно загружены!")

    if 'df' not in st.session_state:
        st.info("Нажмите кнопку «Загрузить данные», чтобы начать.")
        return

    df = st.session_state['df']
    st.subheader("Просмотр данных")
    st.dataframe(df.head())

    st.subheader("Статистика")
    st.write(df.describe())

    if 'models' not in st.session_state:
        with st.spinner("Выполняется предобработка и обучение моделей..."):
            X, y, encoders, feature_names = preprocess_data(df)
            numerical_cols = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyPay',
                              'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialCaseEstimate',
                              'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler, X_train, X_test = scale_features(X_train, X_test, numerical_cols)

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            }
            trained_models = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                trained_models[name] = model

            st.session_state.update({
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'models': trained_models,
                'encoders': encoders,
                'scaler': scaler,
                'feature_names': feature_names,
                'numerical_cols': numerical_cols
            })
        st.success("Модели обучены!")

    st.header("Выбор модели")
    model_names = list(st.session_state['models'].keys())
    selected_model_name = st.selectbox(
        "Выберите модель для анализа и прогноза",
        model_names,
        index=2
    )
    selected_model = st.session_state['models'][selected_model_name]

    st.header(f"Результаты модели: {selected_model_name}")
    metrics, fig = evaluate_model(
        selected_model,
        st.session_state['X_test'],
        st.session_state['y_test'],
        selected_model_name
    )

    st.metric("MAE", f"{metrics['MAE']:.2f}")
    st.metric("RMSE", f"{metrics['RMSE']:.2f}")
    st.metric("R²", f"{metrics['R2']:.4f}")

    st.pyplot(fig)

    st.header("Важность признаков")
    imp_df = get_feature_importance(
        selected_model,
        st.session_state['feature_names'],
        selected_model_name
    )
    if imp_df is None:
        st.write("Важность признаков недоступна для этой модели.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        top_n = min(10, len(imp_df))
        ax.barh(
            imp_df['feature'][:top_n][::-1],
            imp_df['importance'][:top_n][::-1]
        )
        ax.set_xlabel('Важность')
        ax.set_title(f'{selected_model_name}: Топ-{top_n} важных признаков')
        st.pyplot(fig)

    st.header("Предсказание стоимости возмещения")
    st.write("Введите параметры нового страхового случая:")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Возраст", min_value=13, max_value=76, value=35)
            gender = st.selectbox("Пол", ["M", "F"])
            marital_status = st.selectbox("Семейное положение", ["Single", "Married", "Widowed", "Divorced"])
            dependent_children = st.number_input("Дети на иждивении", min_value=0, value=0)
            dependents_other = st.number_input("Другие иждивенцы", min_value=0, value=0)
            weekly_pay = st.number_input("Еженедельная зарплата ($)", min_value=0, value=500)
            part_time_full_time = st.selectbox("Тип занятости", ["Part Time", "Full Time"])
            hours_per_week = st.number_input("Часов в неделю", min_value=0, max_value=168, value=40)
            days_per_week = st.number_input("Дней в неделю", min_value=1, max_value=7, value=5)
        with col2:
            claim_description = st.text_input("Описание заявки", value="Sprain")
            initial_estimate = st.number_input("Начальная оценка ($)", min_value=0, value=5000)
            accident_date = st.date_input("Дата несчастного случая")
            report_date = st.date_input("Дата сообщения о случае")

        submitted = st.form_submit_button("Предсказать")
        if submitted:
            if 'models' not in st.session_state:
                st.error("Сначала загрузите данные и обучите модели.")
            else:
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Gender': [gender],
                    'MaritalStatus': [marital_status],
                    'DependentChildren': [dependent_children],
                    'DependentsOther': [dependents_other],
                    'WeeklyPay': [weekly_pay],
                    'PartTimeFullTime': [part_time_full_time],
                    'HoursWorkedPerWeek': [hours_per_week],
                    'DaysWorkedPerWeek': [days_per_week],
                    'ClaimDescription': [claim_description],
                    'InitialCaseEstimate': [initial_estimate]
                })
                input_data['AccidentMonth'] = accident_date.month
                input_data['AccidentDayOfWeek'] = accident_date.weekday()
                input_data['ReportingDelay'] = (report_date - accident_date).days

                for col, le in st.session_state['encoders'].items():
                    if col in input_data.columns:
                        val = input_data[col].iloc[0]
                        if val in le.classes_:
                            input_data[col] = le.transform([val])[0]
                        else:
                            input_data[col] = -1

                numerical = st.session_state['numerical_cols']
                input_data[numerical] = st.session_state['scaler'].transform(input_data[numerical])
                input_data = input_data[st.session_state['feature_names']]

                prediction = selected_model.predict(input_data)[0]
                st.success(f"Прогноз ({selected_model_name}): {prediction:,.2f}")


if __name__ == "__main__":
    main()
