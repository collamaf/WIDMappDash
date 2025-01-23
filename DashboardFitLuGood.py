import base64

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from scipy.optimize import curve_fit

# Creiamo un dataset di esempio
#df = sns.load_dataset("penguins").dropna()
matplotlib.pyplot.switch_backend('Agg')


def exponential(x, a, b):
#    return a * np.exp(-x/(6*24))
    return a * np.exp(-x/(b/0.69314))

# Leggi il file CSV
df = pd.read_csv('Lu_IFO_2001_senzacorrezione.csv', parse_dates=['tempo_h'])
#df['tempo_h'] = pd.to_datetime(df['tempo_h'], unit="h")
df=df.set_index('tempo_h')
df['time_delta'] = (df.index - df.index[0]).total_seconds()/60/60
df=df.set_index('time_delta')
#df = df.drop("tempo_h", axis=1)

# Inizializzazione dell'app Dash
app = dash.Dash(__name__)


# Funzione per generare un grafico scatter con Seaborn
def generate_scatterplot(x_range=None):
    plt.figure(figsize=(6, 4))
    for col in df:
        sns.scatterplot(data=df, x=df.index, y=df[col], alpha=0.7, label=col)
    #sns.scatterplot(data=df, x=df.index, y="4_1072", alpha=0.7)
    #sns.scatterplot(data=df, x=df.index, y="0_1057", alpha=0.7)

#    sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", alpha=0.7)
    if x_range:
        #plt.xlim(x_range)
        plt.axvline(x=x_range[0], color="salmon", linestyle="--", linewidth=1, label="Vertical Line")
        plt.axvline(x=x_range[1], color="salmon", linestyle="--", linewidth=1, label="Vertical Line")

    plt.ylabel("Raw Rate [CPS]")
    plt.xlabel("Time [h]")
    plt.legend(title="Sensors",bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.legend(title="Sensors", loc="upper left")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    img = Image.open(buf)
    return img


# Layout dell'app
app.layout = html.Div(
    style={"display": "flex"},
    children=[
        # Grafico A (scatterplot con Seaborn)
        html.Div(
            style={"width": "50%"},
            children=[
                html.H3("Grafico Dati"),
                html.Img(id="scatterplot", style={"width": "100%"}, src=None),
                dcc.RangeSlider(
                    id="range-slider",
                    min=df.index.min(),
                    max=df.index.max(),
                    step=0.1,
                    value=[df.index.min(), df.index.max()],
                    #marks={round(i, 1): f"{i:.1f}" for i in
                    #       np.linspace(df["bill_length_mm"].min(), df["bill_length_mm"].max(), 10)},
                ),
            ],
        ),
        # Grafico B (valori estremi del selettore)
        html.Div(
            style={"width": "50%", "padding": "20px"},
            children=[
                html.H3("Risultato fit"),
                dcc.Graph(id="extreme-values", style={"height": "400px"}),
            ],
        ),
    ],
)


# Callback per aggiornare il grafico scatterplot
@app.callback(
    Output("scatterplot", "src"),
    Input("range-slider", "value"),
)
def update_scatterplot(selected_range):
    img = generate_scatterplot(x_range=selected_range)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    encoded_img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded_img


# Callback per aggiornare il grafico dei valori estremi
@app.callback(
    Output("extreme-values", "figure"),
    Input("range-slider", "value"),
)
def update_extreme_values(selected_range):
    allTaus=[]
    for col in df.columns:
        # Parametri iniziali per il fit
        initial_guess = [1000, 150]
        # Step 3: Eseguire il fitting
        # y_data = df['0_1057']
        # y_data = df[col]
        filtered_df = df[(df.index >= selected_range[0]) & (df.index <= selected_range[1])]
        #params, covariance = curve_fit(exponential, df.index, df[col], p0=initial_guess)
        params, covariance = curve_fit(exponential, filtered_df.index, filtered_df[col], p0=initial_guess)

        # Stampare i parametri del fit
        a, tau = params
        print(f"Parametri del fit: a={a}, b={tau}")
        allTaus.append(tau / 24.0)
        # Step 4: Calcolare i valori previsti dal modello
        y_fit = exponential(df.index, *params)

        # Grafico con Seaborn
    #fig = plt.figure(figsize=(10, 6))
    #ax = sns.scatterplot(x=df.index, y=df["4_1072"], label=f'{1} data')  # Dati originali
        # px.line(df,y=df.columns, title=f"All Rates Raw - {0}", color_discrete_sequence=px.colors.qualitative.Pastel)
        #ax.set(xlabel="Time [h]", ylabel="Uncorrected Rate [CPS]")
        #plt.plot(df.index, y_fit, label=f'{col} fit (a={a:.2f}, tau={tau / 24:.2f})')
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        #plt.tight_layout()
        #plt.savefig("AllDecay.pdf")
    print("CIAO: ", selected_range, allTaus)
    fig = px.bar(
        x=range(10),
        y=allTaus,
        labels={"x": "Channel", "y": "T1/2 [d]"},
        title="T1/2 fittati",
    )
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                x0=-0.5,  # Inizia appena prima della prima barra
                x1=10 - 0.5,  # Termina dopo l'ultima barra
                y0=6.6,  # Altezza della linea
                y1=6.6,  # Altezza della linea
                line=dict(color="salmon", width=2, dash="dash"),  # Stile della linea
            )
        ]
    )
    #fig = px.bar(
    #    x=["Min", "Max"],
    #    y=selected_range,
    #    labels={"x": "Estremo", "y": "Valore"},
    #    title="Valori Estremi del Range Selezionato",
    #)
    return fig


# Esecuzione dell'app
if __name__ == "__main__":
    app.run_server(debug=True)