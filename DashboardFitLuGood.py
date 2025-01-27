import base64

import dash
from dash import Dash, Input, Output, ctx, html, dcc, callback

import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
from scipy.optimize import curve_fit
import random

# Creiamo un dataset di esempio
#df = sns.load_dataset("penguins").dropna()
matplotlib.pyplot.switch_backend('Agg')


def exponential(x, a, b):
    #    return a * np.exp(-x/(6*24))
    return a * np.exp(-x / (b / 0.69314))


def generate_meas_intervals(N, M, L):
    numbers = []
    while len(numbers) < N:
        num = random.uniform(0, M)
        # Check if the number is sufficiently far from all others
        if all(abs(num - existing) >= L for existing in numbers):
            numbers.append(num)
    return numbers


#rate_Lu_IFO_soglie13_Overweekend2401_senzacorrezione
# Leggi il file CSV
df = pd.read_csv('rate_Lu_IFO_soglie13_Overweekend2401_senzacorrezione.csv', parse_dates=['tempo_h'])
#df = pd.read_csv('Lu_IFO_2001_senzacorrezione.csv', parse_dates=['tempo_h'])
#df['tempo_h'] = pd.to_datetime(df['tempo_h'], unit="h")
df = df.set_index('tempo_h')
df['time_delta'] = (df.index - df.index[0]).total_seconds() / 60 / 60
df = df.set_index('time_delta')
#df = df.drop("tempo_h", axis=1)
df_to_show = df

total_duration_min = len(df_to_show) * 100 / 60

numberOfMeas = 10
measDurationInMin = 30
minDistanceBetweenMeasInMin = 180


generated_measurements = generate_meas_intervals(numberOfMeas, total_duration_min, minDistanceBetweenMeasInMin)

print(
    f"Caricato dataset: è lungo {len(df_to_show)} misure, quindi {total_duration_min:.0f}min, {len(df_to_show) / 3600 * 100:.1f}h")

for meas in generated_measurements:
    print(f"Misura generata: {meas:.1f} min")
# Inizializzazione dell'app Dash
app = dash.Dash(__name__)


# Funzione per generare un grafico scatter con Seaborn
def generate_scatterplot(x_range=None, selected_fraction=None):
    #print("entro in generate_scatterplot", x_range, selected_fraction)
    plt.figure(figsize=(6, 4))
    if selected_fraction:
        df_to_show = df.sample(frac=selected_fraction / 100, random_state=42)
    else:
        df_to_show = df
    for col in df:
        sns.scatterplot(data=df_to_show, x=df_to_show.index, y=df_to_show[col], alpha=0.7, label=col)
    #sns.scatterplot(data=df, x=df.index, y="4_1072", alpha=0.7)
    #sns.scatterplot(data=df, x=df.index, y="0_1057", alpha=0.7)

    #    sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", alpha=0.7)
    if x_range:
        #plt.xlim(x_range)
        plt.axvline(x=x_range[0], color="salmon", linestyle="--", linewidth=1, label="_nolegend_")
        plt.axvline(x=x_range[1], color="salmon", linestyle="--", linewidth=1, label="_nolegend_")

    for meas in generated_measurements:
        #plt.axvline(x=meas/60, color="blue", linestyle="--", linewidth=1, label="Vertical Line")
        plt.axvspan((meas-measDurationInMin)/60, (meas+measDurationInMin)/60, color='limegreen', alpha=0.3, label="_nolegend_")

    plt.ylabel("Raw Rate [CPS]")
    plt.xlabel("Time [h]")
    plt.legend(title="Sensors", bbox_to_anchor=(1.05, 1), loc='upper left')
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
    style={"display": "flex", "flexDirection": "column", "alignItems": "center",
           "fontFamily": "Helvetica, Arial, sans-serif", },
    children=[
        # Titolo centrato in alto
        html.H1(
            "WIDMApp",
            style={
                "textAlign": "center",
                "width": "100%",
                "padding": "20px",
                "marginBottom": "20px",
                "fontSize": "50px",
            },
        ),
        html.Div(
            style={"display": "flex", "width": "100%"},
            children=[
                # Grafico A (scatterplot con Seaborn)
                html.Div(
                    style={"width": "50%", "padding": "20px"},
                    children=[
                        html.H3("Grafico Dati"),
                        html.Img(id="scatterplot", style={"width": "100%"}, src=None),
                        html.Div(
                            children=[
                                html.H4("Selezione Range"),
                                dcc.RangeSlider(
                                    id="range-slider",
                                    min=df.index.min(),
                                    max=df.index.max(),
                                    step=0.1,
                                    value=[df.index.min(), df.index.max()],
                                    marks=None,
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.H4("Selezione Frazione", id="fraction-title"),
                                dcc.Slider(
                                    id="fraction-slider",
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=100,
                                    marks={i: str(i) for i in range(0, 101, 2)},
                                ),
                            ]
                        ),
                    ],
                ),
                # Grafico B (valori estremi del selettore)
                html.Div(
                    style={"width": "50%", "padding": "20px"},
                    children=[
                        html.H3("Risultato fit"),
                        dcc.Graph(id="all-taus", style={"height": "400px"}),
                        dcc.Graph(id="all-taus-diff", style={"height": "400px"}),
                    ],
                ),
            ],
        ),
    ],
)


# app.layout = html.Div(
#     style={"display": "flex"},
#     children=[
#         # Grafico A (scatterplot con Seaborn)
#         html.H1("WIDMApp"),
#         html.Div(
#             style={"width": "50%"},
#             children=[
#                 html.H3("Grafico Dati"),
#                 html.Img(id="scatterplot", style={"width": "100%"}, src=None),
#                 html.Div(
#                     children=[
#                         html.H4("Selezione Range"),
#                         dcc.RangeSlider(
#                             id="range-slider",
#                             min=df.index.min(),
#                             max=df.index.max(),
#                             step=0.1,
#                             value=[df.index.min(), df.index.max()],
#                             marks=None,
#                             #marks={round(i, 1): f"{i:.1f}" for i in
#                             #       np.linspace(df["bill_length_mm"].min(), df["bill_length_mm"].max(), 10)},
#                         ),
#                     ]
#                 ),
#                 html.Div(
#                     children=[
#                         html.H4("Selezione Frazione"),
#                         dcc.Slider(
#                             id="fraction-slider",
#                             min=0,
#                             max=100,
#                             step=1,
#                             value=100,
#                             marks = {i: str(i) for i in range(0, 101, 2)},
#                             # marks={round(i, 1): f"{i:.1f}" for i in
#                             #       np.linspace(df["bill_length_mm"].min(), df["bill_length_mm"].max(), 10)},
#                         ),
#                     ]
#                 ),
#
#             ],
#         ),
#         # Grafico B (valori estremi del selettore)
#         html.Div(
#             style={"width": "50%", "padding": "20px"},
#             children=[
#                 html.H3("Risultato fit"),
#                 dcc.Graph(id="all-taus", style={"height": "400px"}),
#                 dcc.Graph(id="all-taus-diff", style={"height": "400px"}),
#             ],
#         ),
#     ],
# )


# Callback per aggiornare il titolo H4
@app.callback(
    Output("fraction-title", "children"),
    Input("fraction-slider", "value"),
)
def update_title(slider_value):
    return f"Selezione Frazione: {slider_value}"  # Testo con valore dinamico


# Callback per aggiornare il grafico scatterplot
@app.callback(
    Output("scatterplot", "src"),
    Input("range-slider", "value"),
    Input("fraction-slider", "value"),
)
def update_scatterplot(selected_range, selected_fraction):
    triggered_id = ctx.triggered_id
    #print("AAAA", triggered_id)
    img = generate_scatterplot(x_range=selected_range, selected_fraction=selected_fraction)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    encoded_img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded_img


# Callback per aggiornare il grafico dei valori estremi
@app.callback(
    Output("all-taus", "figure"),
    Input("range-slider", "value"),
    Input("fraction-slider", "value"),
)
def update_all_taus_graph(selected_range, selected_fraction):
    allTaus = []
    if selected_fraction:
        df_to_show = df.sample(frac=selected_fraction / 100, random_state=42)
    else:
        df_to_show = df
    #print("Fitto dataset lungo: ", len(df_to_show))
    for col in df.columns:
        # Parametri iniziali per il fit
        initial_guess = [1000, 150]
        # Step 3: Eseguire il fitting
        # y_data = df['0_1057']
        # y_data = df[col]
        filtered_df = df_to_show[(df_to_show.index >= selected_range[0]) & (df_to_show.index <= selected_range[1])]
        #params, covariance = curve_fit(exponential, df.index, df[col], p0=initial_guess)
        params, covariance = curve_fit(exponential, filtered_df.index, filtered_df[col], p0=initial_guess)

        # Stampare i parametri del fit
        a, tau = params
        #print(f"Parametri del fit: a={a}, b={tau}")
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
    #print("CIAO: ", selected_range, allTaus)
    # channels = range(10)
    # data = pd.DataFrame({"Channel": channels, "T1/2 [d]": allTaus})
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=data, x="Channel", y="T1/2 [d]", palette="viridis")
    #
    # # Personalizzazione
    # plt.title("T1/2 fittati", fontsize=14)
    # plt.xlabel("Channel", fontsize=12)
    # plt.ylabel("T1/2 [d]", fontsize=12)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.tight_layout()
    # fig = plt.gcf()
    #
    #fig, axs = plt.subplots(nrows=2, ncols=1,figsize=(10, 6))
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


#Callback per aggiornare il grafico dei valori estremi
@app.callback(
    Output("all-taus-diff", "figure"),
    Input("range-slider", "value"),
    Input("fraction-slider", "value"),
)
def update_all_taus_graph(selected_range, selected_fraction):
    allTausDiff = []
    if selected_fraction:
        df_to_show = df.sample(frac=selected_fraction / 100, random_state=42)
    else:
        df_to_show = df
    print("Fitto dataset lungo: ", len(df_to_show))
    for col in df.columns:
        # Parametri iniziali per il fit
        initial_guess = [1000, 150]
        # Step 3: Eseguire il fitting
        # y_data = df['0_1057']
        # y_data = df[col]
        filtered_df = df_to_show[(df_to_show.index >= selected_range[0]) & (df_to_show.index <= selected_range[1])]
        #params, covariance = curve_fit(exponential, df.index, df[col], p0=initial_guess)
        params, covariance = curve_fit(exponential, filtered_df.index, filtered_df[col], p0=initial_guess)

        # Stampare i parametri del fit
        a, tau = params
        #print(f"Parametri del fit: a={a}, b={tau}")
        allTausDiff.append((tau / 24.0 - 6.6) / 6.6 * 100)
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
    #print("CIAO: ", selected_range, allTausDiff)
    # channels = range(10)
    # data = pd.DataFrame({"Channel": channels, "T1/2 [d]": allTaus})
    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=data, x="Channel", y="T1/2 [d]", palette="viridis")
    #
    # # Personalizzazione
    # plt.title("T1/2 fittati", fontsize=14)
    # plt.xlabel("Channel", fontsize=12)
    # plt.ylabel("T1/2 [d]", fontsize=12)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.tight_layout()
    # fig = plt.gcf()
    #
    #fig, axs = plt.subplots(nrows=2, ncols=1,figsize=(10, 6))
    fig = px.bar(
        x=range(10),
        y=allTausDiff,
        labels={"x": "Channel", "y": "Diff T1/2 [%]"},
        title="Errori % su T1/2",
    )

    fig.update_layout(
        #xaxis=dict(range=[0, 15]),  # Range fisso per l'asse X
        yaxis=dict(range=[-100, 100]),  # Range fisso per l'asse Y
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
