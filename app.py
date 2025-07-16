import os
import io
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from flask import Flask, render_template, request, send_file, redirect, url_for

app = Flask(__name__)

def bass_model_vectorized(t, p, q, m):
    adoption_cum = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
    adoption_new = np.diff(np.insert(adoption_cum, 0, 0))
    return adoption_new

def logistic_growth(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

def log_logistic_growth(t, K, alpha, beta):
    return K / (1 + (t / alpha) ** (-beta))

def simulate_markov(initial_state, transitions, steps):
    states = [initial_state]
    for _ in range(steps - 1):
        states.append(states[-1] @ transitions)
    return np.array(states)

def generate_month_labels(start_date, num_months):
    turkce_aylar = [
        "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
        "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"
    ]
    tarih_listesi = []
    for i in range(num_months):
        tarih = start_date + relativedelta(months=i)
        ay_isim = turkce_aylar[tarih.month - 1]
        tarih_listesi.append(f"{ay_isim} {tarih.year}")
    return tarih_listesi

def create_projection_graphs(df):
    static_path = os.path.join('static', 'graphs')
    os.makedirs(static_path, exist_ok=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Seaborn tema ayarı
    sns.set_theme(style="whitegrid")

    def disable_sci_format():
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        plt.ticklabel_format(style='plain', axis='y')

    def finalize_plot(title, ylabel, filename):
        disable_sci_format()
        plt.xlabel("Ay")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(os.path.join(static_path, filename))
        plt.close()

    # Kullanıcı Grafiği
    plt.figure(figsize=(14, 6))
    sns.lineplot(x="Ay", y="Kullanici (Bass)", data=df, label="Kullanıcı (Bass)", marker='o', color='green')
    sns.lineplot(x="Ay", y="Kullanici (Logistic)", data=df, label="Kullanıcı (Logistic)", linestyle='--', color='blue')
    sns.lineplot(x="Ay", y="Kullanici (Log-Logistic)", data=df, label="Kullanıcı (Log-Logistic)", linestyle=':', color='orange')
    finalize_plot("Kullanıcı Projeksiyonu", "Kullanıcı Sayısı", 'kullanici_projeksiyon.png')

    # Kümülatif İçerik Grafiği
    plt.figure(figsize=(14, 6))
    sns.lineplot(x="Ay", y="Icerik (Bass)", data=df, label="İçerik (Bass)", color='darkgreen')
    sns.lineplot(x="Ay", y="Icerik (Poisson) Kümülatif", data=df, label="İçerik (Poisson)", linestyle='--', color='blue')
    sns.lineplot(x="Ay", y="Icerik (Lineer) Kümülatif", data=df, label="İçerik (Lineer)", linestyle='-.', color='purple')
    sns.lineplot(x="Ay", y="Icerik (Log-Logistic) Kümülatif", data=df, label="İçerik (Log-Logistic)", linestyle=':', color='orange')
    finalize_plot("Kümülatif İçerik Projeksiyonu", "Toplam İçerik", 'icerik_kumulatif_projeksiyon.png')

    # Aylık İçerik Tüm Modeller Tek Grafik
    plt.figure(figsize=(14, 6))
    sns.lineplot(x="Ay", y="Icerik Aylik (Bass)", data=df, label="İçerik Aylık (Bass)", marker='o')
    sns.lineplot(x="Ay", y="Icerik Aylik (Poisson)", data=df, label="İçerik Aylık (Poisson)", linestyle='--')
    sns.lineplot(x="Ay", y="Icerik Aylik (Lineer)", data=df, label="İçerik Aylık (Lineer)", linestyle='-.')
    sns.lineplot(x="Ay", y="Icerik Aylik (Log-Logistic)", data=df, label="İçerik Aylık (Log-Logistic)", linestyle=':')
    finalize_plot("Aylık İçerik Projeksiyonu - Tüm Modeller", "Aylık İçerik Sayısı", 'icerik_aylik_tum_modeller.png')

projection_df_global = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global projection_df_global
    projection_df = None
    error = None
    form_values = {
        'market_size': '',
        'p': '',
        'q': '',
        'writer_ratio': '',
        'daily_posts': '',
        'initial_users': '0',
        'initial_content': '0',
        'content_scenario': 'realistic',
        'proj_months': '12'
    }

    if request.method == 'POST':
        try:
            form_values.update({
                'market_size': request.form['market_size'],
                'p': request.form['p'],
                'q': request.form['q'],
                'writer_ratio': request.form['writer_ratio'],
                'daily_posts': request.form['daily_posts'],
                'initial_users': request.form['initial_users'],
                'initial_content': request.form['initial_content'],
                'content_scenario': request.form.get('content_scenario', 'realistic'),
                'proj_months': request.form.get('proj_months', '12'),
            })

            m = int(form_values['market_size'])
            p = float(form_values['p']) / 100
            q = float(form_values['q']) / 100
            writer_ratio = float(form_values['writer_ratio']) / 100
            daily_posts = float(form_values['daily_posts'])
            initial_users = int(form_values['initial_users'])
            initial_content = int(form_values['initial_content'])
            proj_months = int(form_values['proj_months'])

            months = np.arange(1, proj_months + 1)
            monthly_post_rate = daily_posts * 30

            bass_new = bass_model_vectorized(months, p, q, m)
            bass_cum = np.cumsum(bass_new) + initial_users
            logistic = logistic_growth(months, m, 0.5, proj_months / 2) + initial_users
            log_logistic = log_logistic_growth(months, m, proj_months / 2, 2) + initial_users

            scale_factor = {
                'optimistic': 1.2,
                'pessimistic': 0.7,
                'realistic': 1.0
            }.get(form_values['content_scenario'], 1.0)

            bass_cum_scenario = bass_cum * scale_factor
            logistic_scenario = logistic * scale_factor
            log_logistic_scenario = log_logistic * scale_factor

            transitions = np.array([
                [0.85, 0.1, 0.05],
                [0.05, 0.75, 0.20],
                [0.02, 0.03, 0.95]
            ])

           
            initial_state = np.array([1.0, 0.0, 0.0])
            markov_states = simulate_markov(initial_state, transitions, proj_months)

            bass_content_monthly = np.zeros(proj_months)
            poisson_content_monthly = np.zeros(proj_months)
            linear_content_monthly = np.zeros(proj_months)
            log_logistic_content_monthly = np.zeros(proj_months)

            for i in range(proj_months):
                churn_factor = markov_states[i, 0]
                bass_content_monthly[i] = bass_cum_scenario[i] * writer_ratio * monthly_post_rate * churn_factor
                poisson_content_monthly[i] = logistic_scenario[i] * writer_ratio * monthly_post_rate * churn_factor 
                linear_content_monthly[i] = logistic_scenario[i] * writer_ratio * monthly_post_rate * churn_factor  * 0.9
                log_logistic_content_monthly[i] = log_logistic_scenario[i] * writer_ratio * monthly_post_rate * churn_factor

            bass_content_cum = initial_content + np.cumsum(bass_content_monthly)
            poisson_content_cum = initial_content + np.cumsum(poisson_content_monthly)
            linear_content_cum = initial_content + np.cumsum(linear_content_monthly)
            log_logistic_content_cum = initial_content + np.cumsum(log_logistic_content_monthly)

            start_date = datetime(2025, 9, 1)
            month_labels = generate_month_labels(start_date, proj_months)

            projection_df = pd.DataFrame({
                "Ay": month_labels,
                "Kullanici (Bass)": bass_cum_scenario.astype(int),
                "Kullanici (Logistic)": logistic_scenario.astype(int),
                "Kullanici (Log-Logistic)": log_logistic_scenario.astype(int),
                "Icerik (Bass)": bass_content_cum.astype(int),
                "Icerik (Poisson) Kümülatif": poisson_content_cum.astype(int),
                "Icerik (Lineer) Kümülatif": linear_content_cum.astype(int),
                "Icerik (Log-Logistic) Kümülatif": log_logistic_content_cum.astype(int),
                "Aktif (%)": (markov_states[:, 0] * 100).round(1),
                "Churn (%)": (markov_states[:, 2] * 100).round(1),
                # Aylık içerik kolonları - yeni eklendi
                "Icerik Aylik (Bass)": bass_content_monthly.astype(int),
                "Icerik Aylik (Poisson)": poisson_content_monthly.astype(int),
                "Icerik Aylik (Lineer)": linear_content_monthly.astype(int),
                "Icerik Aylik (Log-Logistic)": log_logistic_content_monthly.astype(int),
            })

            projection_df_global = projection_df.copy()
            create_projection_graphs(projection_df)

        except Exception as e:
            error = str(e)
            projection_df = None

    return render_template('index.html', table=projection_df, error=error, form_values=form_values)

@app.route('/download_excel')
def download_excel():
    global projection_df_global
    if projection_df_global is None:
        return redirect(url_for('index'))

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        projection_df_global.to_excel(writer, index=False, sheet_name='Projeksiyon')
    output.seek(0)

    return send_file(output, download_name="projeksiyon.xlsx", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
