# Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm
Elbette, iki proje için ayrı ayrı README dosyaları oluşturabilirim. İşte her iki proje için README dosyalarının örnekleri:

### Proje 1: Turizm Geliri Tahmini
# Turizm Geliri Tahmini

Bu proje, Türkiye'nin turizm gelirlerini yıllara göre tahmin etmek için çeşitli zaman serisi modelleme tekniklerini kullanır. Proje, turizm geliri verilerini kullanarak dört farklı modelin performansını değerlendirir: ARIMA, SARIMA, Holt-Winters ve Prophet. Bu proje Tubitak 2209 tarafından desteklenmektedir. Model sonuçları bir arayüze eklenmiş olup 4 modelinde değerlerini başarı metriklerini ve test veri seti için yaptığı tahminleri görüntüleyebilirsiniz. Aynı zamanda metrikleri görselleştirirseniz. Bu görselleri her bir metriğin birbiri ile kıyaslanmasını görseller ile bize aktarır.


## İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Modeller](#modeller)
- [Metrikler](#metrikler)
- [Görselleştirme](#görselleştirme)

## Proje Hakkında
Bu projede, Türkiye'nin turizm gelirlerini yıllara göre tahmin ederek gelecekteki gelirleri öngörmeyi amaçladık. Verileri kullanarak dört farklı modelin performansını değerlendirdik ve sonuçları görselleştirerek karşılaştırdık.

## Veri Seti
Veri seti Tursab'ın [turizm istatistikleri](https://www.tursab.org.tr/turkiye-turizm-istatistikleri) sitesinden alınmıştır. Bu veri seti, yıllara göre Türkiye'nin turizm gelirlerini içermektedir.



## Kurulum
Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. **Depoyu Klonlayın:**
```markdown

   ```sh
   git clone https://github.com/kullanici_adi/Turizm_Geliri_Tahmini.git
   cd Turizm_Geliri_Tahmini
   ```

2. **Sanal Ortam Oluşturun ve Aktif Hale Getirin:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # MacOS/Linux
   venv\Scripts\activate  # Windows
   ```

3. **Gerekli Bağımlılıkları Yükleyin:**
   ```sh
   pip install -r requirements.txt
   ```

## Kullanım
Projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. **Streamlit Uygulamasını Başlatın:**
   ```sh
   streamlit run app.py
   ```

2. **Model Seçimi:**
   Streamlit arayüzünden bir model seçin ve "Modeli Değerlendir" butonuna tıklayın.
   <img width="850" alt="Ekran Resmi 2024-07-06 15 13 38" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/adffc4e9-b078-4bab-b2d9-1b652c13355d">


4. **Metrikleri Görselleştirin:**
   Metrikleri görselleştirmek için bir metrik seçin ve "Metrikleri Görselleştir" butonuna tıklayın.
   <img width="979" alt="Ekran Resmi 2024-07-06 15 15 22" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/6da5208e-5d3b-46dd-b23b-126968ea06aa">


## Modeller
Projede kullanılan modeller:
- **ARIMA:** AutoRegressive Integrated Moving Average
- **SARIMA:** Seasonal AutoRegressive Integrated Moving Average
- **Holt-Winters:** Holt-Winters Exponential Smoothing
- **Prophet:** Facebook Prophet

## Metrikler
Model performansını değerlendirmek için kullanılan metrikler:
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error
- **sMAPE:** Symmetric Mean Absolute Percentage Error
- **ME:** Mean Error

## Görselleştirme
Model performans metriklerini görselleştirmek için bir metrik seçin ve "Metrikleri Görselleştir" butonuna tıklayın. Seçilen metrik için her modelin performansını bar grafiği olarak görüntüleyebilirsiniz.

<img width="850" alt="Ekran Resmi 2024-07-06 15 13 38" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/c553b488-78c2-44e7-b7fb-6461cac61fc9">


<img width="755" alt="Ekran Resmi 2024-07-06 15 15 52" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/cf9fbf45-bf31-4ae9-ac1f-75f53e68b562">

Elbette, işte README dosyanızın İngilizce çevirisi:

# Customer Revenue Prediction with Machine Learning Models in Tourism

This project uses various time series modeling techniques to predict Turkey's tourism revenues by year. The project evaluates the performance of four different models using tourism revenue data: ARIMA, SARIMA, Holt-Winters, and Prophet. This project is supported by Tubitak 2209. The model results are integrated into an interface where you can view the performance metrics and predictions for the test dataset for all four models. Additionally, you can visualize the metrics to compare the performance of each metric across models through graphs.

## Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Metrics](#metrics)
- [Visualization](#visualization)

## About the Project
In this project, we aim to predict Turkey's tourism revenues by year to forecast future revenues. We evaluated the performance of four different models using the data and visualized the results for comparison.

## Dataset
The dataset was obtained from Tursab's [tourism statistics](https://www.tursab.org.tr/turkiye-turizm-istatistikleri) site. This dataset contains Turkey's tourism revenues by year.

## Installation
Follow these steps to set up and run the project:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your_username/Tourism_Revenue_Prediction.git
   cd Tourism_Revenue_Prediction
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # MacOS/Linux
   venv\Scripts\activate  # Windows
   ```

3. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage
Follow these steps to run the project:

1. **Start the Streamlit application:**
   ```sh
   streamlit run app.py
   ```

2. **Model Selection:**
   Select a model from the Streamlit interface and click the "Evaluate Model" button.
   <img width="850" alt="Model Selection" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/adffc4e9-b078-4bab-b2d9-1b652c13355d">

3. **Visualize Metrics:**
   Select a metric to visualize and click the "Visualize Metrics" button.
   <img width="979" alt="Metrics Visualization" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/6da5208e-5d3b-46dd-b23b-126968ea06aa">

## Models
The models used in the project are:
- **ARIMA:** AutoRegressive Integrated Moving Average
- **SARIMA:** Seasonal AutoRegressive Integrated Moving Average
- **Holt-Winters:** Holt-Winters Exponential Smoothing
- **Prophet:** Facebook Prophet

## Metrics
The metrics used to evaluate model performance are:
- **MAE:** Mean Absolute Error
- **MSE:** Mean Squared Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error
- **sMAPE:** Symmetric Mean Absolute Percentage Error
- **ME:** Mean Error

## Visualization
To visualize model performance metrics, select a metric and click the "Visualize Metrics" button. You can view the performance of each model in a bar chart for the selected metric.

<img width="850" alt="Visualization Example 1" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/c553b488-78c2-44e7-b7fb-6461cac61fc9">

<img width="755" alt="Visualization Example 2" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/cf9fbf45-bf31-4ae9-ac1f-75f53e68b562">

<img width="755" alt="Visualization Example 3" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/b2b71709-0ede-4002-89d9-04e60c37fa20">




<img width="755" alt="Ekran Resmi 2024-07-06 15 15 52" src="https://github.com/muratcan77/Musteri_Getiri_Tahmini_Makine_Ogrenmesi_Modelleri_Turizm/assets/60244376/b2b71709-0ede-4002-89d9-04e60c37fa20">


