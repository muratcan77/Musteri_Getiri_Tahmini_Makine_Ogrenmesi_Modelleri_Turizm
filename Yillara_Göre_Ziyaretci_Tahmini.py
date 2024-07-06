import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data
file_path = '/Users/muratcanaltintas/PycharmProjects/ileri_zaman_serisi/yillara_göre_ziyaretci.xlsx'
data = pd.read_excel(file_path)

# Data preprocessing
data.columns = ['Yil', 'Ziyaretci_Sayisi']
egitim_verisi = data[data['Yil'] < 2015]
test_verisi = data[data['Yil'] >= 2015]
test_verisi_metrikler = test_verisi[~test_verisi['Yil'].isin([2020, 2021])]

# Model training and forecasting function
def model_egit_ve_tahmin_et(model_adi):
    if model_adi == 'ARIMA':
        model = ARIMA(egitim_verisi['Ziyaretci_Sayisi'], order=(5, 1, 0))
        model_fit = model.fit()
        tahminler = model_fit.forecast(steps=len(test_verisi))
    elif model_adi == 'SARIMA':
        model = SARIMAX(egitim_verisi['Ziyaretci_Sayisi'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        tahminler = model_fit.forecast(steps=len(test_verisi))
    elif model_adi == 'Holt-Winters':
        mevsimsel_periyotlar = 12
        if len(egitim_verisi) >= 2 * mevsimsel_periyotlar:
            model = ExponentialSmoothing(egitim_verisi['Ziyaretci_Sayisi'], seasonal='add', seasonal_periods=mevsimsel_periyotlar)
            model_fit = model.fit()
            tahminler = model_fit.forecast(steps=len(test_verisi))
        else:
            tahminler = np.nan * np.ones(len(test_verisi))
            st.warning("Holt-Winters mevsimsel modeli için yeterli veri yok.")
    elif model_adi == 'Prophet':
        prophet_verisi = egitim_verisi[['Yil', 'Ziyaretci_Sayisi']].rename(columns={'Yil': 'ds', 'Ziyaretci_Sayisi': 'y'})
        model = Prophet(yearly_seasonality=True, daily_seasonality=False)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(prophet_verisi)
        gelecek = model.make_future_dataframe(periods=len(test_verisi), freq='Y')
        tahmin = model.predict(gelecek)
        tahminler = tahmin['yhat'].iloc[-len(test_verisi):].values
    return tahminler

# Function to calculate performance metrics
def metrikleri_hesapla(test_verisi, tahminler):
    test_verisi_filtered = test_verisi[~test_verisi['Yil'].isin([2020, 2021])]
    tahminler_filtered = tahminler[~test_verisi['Yil'].isin([2020, 2021])]
    mae = mean_absolute_error(test_verisi_filtered['Ziyaretci_Sayisi'], tahminler_filtered)
    mse = mean_squared_error(test_verisi_filtered['Ziyaretci_Sayisi'], tahminler_filtered)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(test_verisi_filtered['Ziyaretci_Sayisi'], tahminler_filtered) * 100
    smape = 100 / len(test_verisi_filtered) * np.sum(2 * np.abs(tahminler_filtered - test_verisi_filtered['Ziyaretci_Sayisi']) / (
                np.abs(test_verisi_filtered['Ziyaretci_Sayisi']) + np.abs(tahminler_filtered)))
    me = np.mean(test_verisi_filtered['Ziyaretci_Sayisi'] - tahminler_filtered)
    return mae, mse, rmse, mape, smape, me

# Streamlit application
st.title('Ziyaretçi Sayısı Tahmini')

model_secimleri = ['ARIMA', 'SARIMA', 'Holt-Winters', 'Prophet']
secili_model = st.selectbox('Bir model seçin', model_secimleri)

if st.button('Modeli Değerlendir'):
    tahminler = model_egit_ve_tahmin_et(secili_model)
    test_verisi['Tahminler'] = tahminler

    if not test_verisi['Tahminler'].isna().all():
        mae, mse, rmse, mape, smape, me = metrikleri_hesapla(test_verisi, test_verisi['Tahminler'])
        st.write(f"**{secili_model} Modeli**")
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")
        st.write(f"RMSE: {rmse}")
        st.write(f"MAPE: {mape}")
        st.write(f"sMAPE: {smape}")
        st.write(f"ME: {me}")

        st.write('Tahminler:')
        test_verisi_display = test_verisi[['Yil', 'Ziyaretci_Sayisi', 'Tahminler']]
        test_verisi_display['Yil'] = test_verisi_display['Yil'].astype(str)
        st.write(test_verisi_display)
    else:
        st.write("Model, yetersiz veri nedeniyle değerlendirilemedi.")

# Predictions and metrics for all models
tahminler_dict = {}
metrikler_dict = {}
for model in model_secimleri:
    tahminler = model_egit_ve_tahmin_et(model)
    if not np.isnan(tahminler).all():
        tahminler_dict[model] = tahminler
        metrikler_dict[model] = metrikleri_hesapla(test_verisi, tahminler)

# Visualization for metrics
metrik_secimleri = ['MAE', 'MSE', 'RMSE', 'MAPE', 'sMAPE', 'ME']
secili_metrik = st.selectbox('Bir metrik seçin', metrik_secimleri)

if st.button('Metrikleri Görselleştir'):
    metrik_degerleri = {model: metrikler_dict[model][metrik_secimleri.index(secili_metrik)] for model in model_secimleri
                        if model in metrikler_dict}

    # Plotting the metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrik_degerleri.keys(), metrik_degerleri.values(), color='skyblue')
    ax.set_title(f'{secili_metrik} Değerlerinin Modeller Arasında Karşılaştırması', fontsize=14)
    ax.set_xlabel('Modeller', fontsize=12)
    ax.set_ylabel(secili_metrik, fontsize=12)
    ax.set_ylim(0, max(metrik_degerleri.values()) * 1.2)
    ax.yaxis.grid(True, linestyle='--', which='both', color='gray', alpha=0.7)
    ax.set_axisbelow(True)

    # Adding metric values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)

    st.pyplot(fig)
