import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, send_file, redirect, url_for
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import io
import os

app = Flask(__name__)

# Bass Difüzyon Modeli
def bass_model_vectorized(t, p, q, m):
    adoption_cum = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
    adoption_new = np.diff(np.insert(adoption_cum, 0, 0))
    return adoption_new

# Lojistik Büyüme
def logistic_growth(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

# Log-Logistik Büyüme
def log_logistic_growth(t, K, alpha, beta):
    return K / (1 + (t / alpha) ** (-beta))

# Markov Geçiş Simülasyonu
def simulate_markov(initial_state, transitions, steps):
    states = [initial_state]
    for _ in range(steps - 1):
        states.append(states[-1] @ transitions)
    return np.array(states)

# Grafik Oluşturucu
def create_projection_graphs(df):
    static_path = os.path.join('static', 'graphs')
    os.makedirs(static_path, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    months_labels = ['Eylül', 'Ekim', 'Kasım', 'Aralık', 'Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran', 'Temmuz', 'Ağustos']

    # Kullanıcı grafiği
    plt.figure(figsize=(12, 7))
    sns.lineplot(x=df["Ay"], y=df["Kullanici (Bass)"], label="Kullanıcı (Bass)", marker='o', linewidth=2)
    sns.lineplot(x=df["Ay"], y=df["Kullanici (Logistic)"], label="Kullanıcı (Logistic)", marker='s', linewidth=2)
    sns.lineplot(x=df["Ay"], y=df["Kullanici (Log-Logistic)"], label="Kullanıcı (Log-Logistic)", marker='^', linewidth=2)
    plt.xlabel("Ay")
    plt.ylabel("Kullanıcı Sayısı")
    plt.title("Kullanıcı Sayısı Projeksiyonu", fontsize=16)
    plt.xticks(df["Ay"], months_labels, rotation=45)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', '.')))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, 'kullanici_projeksiyon.png'))
    plt.close()

    # İçerik grafiği
    plt.figure(figsize=(12, 7))
    sns.lineplot(x=df["Ay"], y=df["Icerik (Bass)"], label="İçerik (Bass)", marker='o', linewidth=2)
    sns.lineplot(x=df["Ay"], y=df["Icerik (Poisson)"], label="İçerik (Poisson)", marker='s', linewidth=2)
    sns.lineplot(x=df["Ay"], y=df["Icerik (Lineer)"], label="İçerik (Lineer)", marker='D', linewidth=2)
    sns.lineplot(x=df["Ay"], y=df["Icerik (Log-Logistic)"], label="İçerik (Log-Logistic)", marker='^', linewidth=2)
    plt.xlabel("Ay")
    plt.ylabel("İçerik Sayısı")
    plt.title("İçerik Sayısı Projeksiyonu", fontsize=16)
    plt.xticks(df["Ay"], months_labels, rotation=45)
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', '.')))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, 'icerik_projeksiyon.png'))
    plt.close()

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
        'content_scenario': 'realistic'
    }

    if request.method == 'POST':
        try:
            form_values['market_size'] = request.form['market_size']
            form_values['p'] = request.form['p']
            form_values['q'] = request.form['q']
            form_values['writer_ratio'] = request.form['writer_ratio']
            form_values['daily_posts'] = request.form['daily_posts']
            form_values['initial_users'] = request.form['initial_users']
            form_values['initial_content'] = request.form['initial_content']
            form_values['content_scenario'] = request.form.get('content_scenario', 'realistic')

            m = int(form_values['market_size'])
            p = float(form_values['p']) / 100
            q = float(form_values['q']) / 100
            writer_ratio = float(form_values['writer_ratio']) / 100
            daily_posts = float(form_values['daily_posts'])
            initial_users = int(form_values['initial_users'])
            initial_content = int(form_values['initial_content'])

            months = np.arange(1, 13)
            monthly_post_rate = daily_posts * 30

            bass_new = bass_model_vectorized(months, p, q, m)
            bass_cum = np.cumsum(bass_new) + initial_users

            logistic = logistic_growth(months, m, 0.5, 6) + initial_users
            log_logistic = log_logistic_growth(months, m, 6, 2) + initial_users

            if form_values['content_scenario'] == 'optimistic':
                scale_factor = 1.2
            elif form_values['content_scenario'] == 'pessimistic':
                scale_factor = 0.7
            else:
                scale_factor = 1.0

            bass_cum_scenario = bass_cum * scale_factor
            logistic_scenario = logistic * scale_factor
            log_logistic_scenario = log_logistic * scale_factor

            bass_content = initial_content + np.cumsum(bass_cum_scenario * writer_ratio * monthly_post_rate)
            poisson_content = initial_content + np.cumsum(logistic_scenario * writer_ratio * monthly_post_rate)
            linear_content = initial_content + np.cumsum(logistic_scenario * writer_ratio * monthly_post_rate * 0.9)
            log_logistic_content = initial_content + np.cumsum(log_logistic_scenario * writer_ratio * monthly_post_rate)

            transitions = np.array([
                [0.85, 0.1, 0.05],
                [0.05, 0.75, 0.20],
                [0.0, 0.0, 1.0]
            ])
            initial_state = np.array([1.0, 0.0, 0.0])
            markov_states = simulate_markov(initial_state, transitions, 12)

            projection_df = pd.DataFrame({
                "Ay": months,
                "Kullanici (Bass)": bass_cum_scenario.astype(int),
                "Kullanici (Logistic)": logistic_scenario.astype(int),
                "Kullanici (Log-Logistic)": log_logistic_scenario.astype(int),
                "Icerik (Bass)": bass_content.astype(int),
                "Icerik (Poisson)": poisson_content.astype(int),
                "Icerik (Lineer)": linear_content.astype(int),
                "Icerik (Log-Logistic)": log_logistic_content.astype(int),
                "Aktif (%)": (markov_states[:, 0] * 100).round(1),
                "Churn (%)": (markov_states[:, 2] * 100).round(1),
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
