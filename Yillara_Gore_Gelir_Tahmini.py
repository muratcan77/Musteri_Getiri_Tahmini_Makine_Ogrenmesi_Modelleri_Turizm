import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import matplotlib.pyplot as plt

# Veriyi yükle
file_path = 'yillara-gore-turkiye-turizm-geliri-1 2 (1).xlsx'
data = pd.read_excel(file_path)

# Veri ön işleme
data.columns = ['Yil', 'Turizm_Geliri', 'Ortalama_Harcama']
data['Ortalama_Harcama'] = data['Ortalama_Harcama'].replace(' ', np.nan).replace('', np.nan).astype(float)
data.dropna(subset=['Ortalama_Harcama'], inplace=True)

# Veriyi eğitim ve test setlerine ayır
egitim_verisi = data[data['Yil'] < 2015]
test_verisi = data[data['Yil'] >= 2015]
# 2020 ve 2021 yıllarını metriklerden çıkar
test_verisi_metrikler = test_verisi[~test_verisi['Yil'].isin([2020, 2021])]


# Modeli eğitme ve tahmin yapma fonksiyonu
def model_egit_ve_tahmin_et(model_adi):
    if model_adi == 'ARIMA':
        model = ARIMA(egitim_verisi['Turizm_Geliri'], order=(5, 1, 0))
        model_fit = model.fit()
        tahminler = model_fit.forecast(steps=len(test_verisi))
    elif model_adi == 'SARIMA':
        model = SARIMAX(egitim_verisi['Turizm_Geliri'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        tahminler = model_fit.forecast(steps=len(test_verisi))
    elif model_adi == 'Holt-Winters':
        mevsimsel_periyotlar = 12
        if len(egitim_verisi) >= 2 * mevsimsel_periyotlar:
            model = ExponentialSmoothing(egitim_verisi['Turizm_Geliri'], seasonal='add',
                                         seasonal_periods=mevsimsel_periyotlar)
            model_fit = model.fit()
            tahminler = model_fit.forecast(steps=len(test_verisi))
        else:
            tahminler = np.nan * np.ones(len(test_verisi))
            st.warning("Holt-Winters mevsimsel modeli için yeterli veri yok.")
    elif model_adi == 'Prophet':
        prophet_verisi = egitim_verisi[['Yil', 'Turizm_Geliri']].rename(columns={'Yil': 'ds', 'Turizm_Geliri': 'y'})
        model = Prophet(yearly_seasonality=True, daily_seasonality=False)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(prophet_verisi)
        gelecek = model.make_future_dataframe(periods=len(test_verisi), freq='Y')
        tahmin = model.predict(gelecek)
        tahminler = tahmin['yhat'].iloc[-len(test_verisi):].values
    return tahminler


# Başarı metriklerini hesaplama fonksiyonu
def metrikleri_hesapla(test_verisi, tahminler):
    mae = mean_absolute_error(test_verisi['Turizm_Geliri'], tahminler)
    mse = mean_squared_error(test_verisi['Turizm_Geliri'], tahminler)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(test_verisi['Turizm_Geliri'], tahminler) * 100
    smape = 100 / len(test_verisi) * np.sum(2 * np.abs(tahminler - test_verisi['Turizm_Geliri']) / (
                np.abs(test_verisi['Turizm_Geliri']) + np.abs(tahminler)))
    me = np.mean(test_verisi['Turizm_Geliri'] - tahminler)
    return mae, mse, rmse, mape, smape, me


# Streamlit uygulaması
st.title('Turizm Geliri Tahmini')

model_secimleri = ['ARIMA', 'SARIMA', 'Holt-Winters', 'Prophet']
secili_model = st.selectbox('Bir model seçin', model_secimleri)

if st.button('Modeli Değerlendir'):
    test_verisi['Tahminler'] = model_egit_ve_tahmin_et(secili_model)

    if not test_verisi['Tahminler'].isna().all():
        mae, mse, rmse, mape, smape, me = metrikleri_hesapla(test_verisi_metrikler, test_verisi['Tahminler'][
            ~test_verisi['Yil'].isin([2020, 2021])])
        st.write(f"**{secili_model} Modeli**")
        st.write(f"MAE: {mae}")
        st.write(f"MSE: {mse}")
        st.write(f"RMSE: {rmse}")
        st.write(f"MAPE: {mape}")
        st.write(f"sMAPE: {smape}")
        st.write(f"ME: {me}")

        st.write('Tahminler:')
        test_verisi_display = test_verisi[['Yil', 'Turizm_Geliri', 'Tahminler']]
        test_verisi_display['Yil'] = test_verisi_display['Yil'].astype(str)
        st.write(test_verisi_display)
    else:
        st.write("Model, yetersiz veri nedeniyle değerlendirilemedi.")

# Tüm modeller için tahminler ve metrikler
tahminler_dict = {}
metrikler_dict = {}
for model in model_secimleri:
    tahminler = model_egit_ve_tahmin_et(model)
    if not np.isnan(tahminler).all():
        tahminler_dict[model] = tahminler
        metrikler_dict[model] = metrikleri_hesapla(test_verisi_metrikler,
                                                   tahminler[~test_verisi['Yil'].isin([2020, 2021])])

# Görselleştirme için butonlar ve grafikler
metrik_secimleri = ['MAE', 'MSE', 'RMSE', 'MAPE', 'sMAPE', 'ME']
secili_metrik = st.selectbox('Bir metrik seçin', metrik_secimleri)

if st.button('Metrikleri Görselleştir'):
    metrik_degerleri = {model: metrikler_dict[model][metrik_secimleri.index(secili_metrik)] for model in model_secimleri
                        if model in metrikler_dict}

    # Grafik oluşturma
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrik_degerleri.keys(), metrik_degerleri.values(), color='skyblue')
    ax.set_title(f'{secili_metrik} Değerlerinin Modeller Arasında Karşılaştırması', fontsize=14)
    ax.set_xlabel('Modeller', fontsize=12)
    ax.set_ylabel(secili_metrik, fontsize=12)
    ax.set_ylim(0, max(metrik_degerleri.values()) * 1.2)  # Y eksenini biraz yukarıdan başlat
    ax.yaxis.grid(True, linestyle='--', which='both', color='gray', alpha=0.7)
    ax.set_axisbelow(True)

    # Barların üzerine metrik değerlerini ekleme
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)

    st.pyplot(fig)
