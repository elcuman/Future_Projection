import matplotlib
matplotlib.use('Agg')  # Arka planda grafik üretimi
from flask import Flask, render_template, request, send_file, redirect, url_for
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Markov Simülasyonu
def simulate_markov(initial_state, transitions, steps):
    states = [initial_state]
    for _ in range(steps - 1):
        states.append(states[-1] @ transitions)
    return np.array(states)

# Grafik Üretici
def create_projection_graphs(df):
    static_path = os.path.join('static', 'graphs')
    os.makedirs(static_path, exist_ok=True)

    # Kullanıcı grafiği
    plt.figure(figsize=(14, 6))
    plt.plot(df["Ay"], df["Kullanici (Bass)"], label="Kullanıcı (Bass)", color='green', marker='o')
    plt.plot(df["Ay"], df["Kullanici (Logistic)"], label="Kullanıcı (Logistic)", color='blue', linestyle='--')
    plt.plot(df["Ay"], df["Kullanici (Log-Logistic)"], label="Kullanıcı (Log-Logistic)", color='orange', linestyle=':')
    plt.xlabel("Ay")
    plt.ylabel("Kullanıcı Sayısı")
    plt.title("Kullanıcı Sayısı Projeksiyonu")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, 'kullanici_projeksiyon.png'))
    plt.close()

    # Aylık İçerik grafiği
    plt.figure(figsize=(14, 6))
    plt.plot(df["Ay"], df["Icerik (Bass) Aylik"], label="İçerik (Bass) Aylık", color='green', marker='o')
    plt.plot(df["Ay"], df["Icerik (Poisson) Aylik"], label="İçerik (Poisson) Aylık", color='blue', linestyle='--')
    plt.plot(df["Ay"], df["Icerik (Lineer) Aylik"], label="İçerik (Lineer) Aylık", color='purple', linestyle='-.')
    plt.plot(df["Ay"], df["Icerik (Log-Logistic) Aylik"], label="İçerik (Log-Logistic) Aylık", color='orange', linestyle=':')
    plt.xlabel("Ay")
    plt.ylabel("Aylık İçerik Sayısı")
    plt.title("Aylık İçerik Sayısı Projeksiyonu")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, 'icerik_aylik_projeksiyon.png'))
    plt.close()

    # Kümülatif İçerik grafiği
    plt.figure(figsize=(14, 6))
    plt.plot(df["Ay"], df["Icerik (Bass) Kümülatif"], label="İçerik (Bass) Kümülatif", color='darkgreen', linestyle='-')
    plt.plot(df["Ay"], df["Icerik (Poisson) Kümülatif"], label="İçerik (Poisson) Kümülatif", color='darkblue', linestyle='--')
    plt.plot(df["Ay"], df["Icerik (Lineer) Kümülatif"], label="İçerik (Lineer) Kümülatif", color='indigo', linestyle='-.')
    plt.plot(df["Ay"], df["Icerik (Log-Logistic) Kümülatif"], label="İçerik (Log-Logistic) Kümülatif", color='darkorange', linestyle=':')
    plt.xlabel("Ay")
    plt.ylabel("Kümülatif İçerik Sayısı")
    plt.title("Kümülatif İçerik Sayısı Projeksiyonu")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(static_path, 'icerik_kumulatif_projeksiyon.png'))
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
        'content_scenario': 'realistic',
        'proj_months': '12'
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
            form_values['proj_months'] = request.form.get('proj_months', '12')

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
                [0.0, 0.0, 1.0]
            ])
            initial_state = np.array([1.0, 0.0, 0.0])
            markov_states = simulate_markov(initial_state, transitions, proj_months)

            # **Aylık içerik hesaplama**
            bass_content_monthly = np.zeros(proj_months)
            poisson_content_monthly = np.zeros(proj_months)
            linear_content_monthly = np.zeros(proj_months)
            log_logistic_content_monthly = np.zeros(proj_months)

            for i in range(proj_months):
                churn_factor = markov_states[i, 0]  # Aktif kullanıcı oranı
                motivasyon = max(0.3, 1 - (0.7 * (i / (proj_months - 1))))  # Motivasyon %30 altına inmesin

                bass_content_monthly[i] = (
                    bass_cum_scenario[i] * writer_ratio * monthly_post_rate * churn_factor * motivasyon
                )
                poisson_content_monthly[i] = (
                    logistic_scenario[i] * writer_ratio * monthly_post_rate * churn_factor * motivasyon
                )
                linear_content_monthly[i] = (
                    logistic_scenario[i] * writer_ratio * monthly_post_rate * churn_factor * motivasyon * 0.9
                )
                log_logistic_content_monthly[i] = (
                    log_logistic_scenario[i] * writer_ratio * monthly_post_rate * churn_factor * motivasyon
                )

            # **Kümülatif içerik hesaplama**
            bass_content = initial_content + np.cumsum(bass_content_monthly)
            poisson_content = initial_content + np.cumsum(poisson_content_monthly)
            linear_content = initial_content + np.cumsum(linear_content_monthly)
            log_logistic_content = initial_content + np.cumsum(log_logistic_content_monthly)

            projection_df = pd.DataFrame({
                "Ay": months,
                "Kullanici (Bass)": bass_cum_scenario.astype(int),
                "Kullanici (Logistic)": logistic_scenario.astype(int),
                "Kullanici (Log-Logistic)": log_logistic_scenario.astype(int),
                "Icerik (Bass) Aylik": bass_content_monthly.astype(int),
                "Icerik (Poisson) Aylik": poisson_content_monthly.astype(int),
                "Icerik (Lineer) Aylik": linear_content_monthly.astype(int),
                "Icerik (Log-Logistic) Aylik": log_logistic_content_monthly.astype(int),
                "Icerik (Bass) Kümülatif": bass_content.astype(int),
                "Icerik (Poisson) Kümülatif": poisson_content.astype(int),
                "Icerik (Lineer) Kümülatif": linear_content.astype(int),
                "Icerik (Log-Logistic) Kümülatif": log_logistic_content.astype(int),
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
