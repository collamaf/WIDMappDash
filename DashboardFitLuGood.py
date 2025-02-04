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

matplotlib.pyplot.switch_backend('Agg')


def exponential(x, a, b):
    #    return a * np.exp(-x/(6*24))
    return a * np.exp(-x / (b / 0.69314))


def perform_all_fits(all_found_taus, selected_range):
    print("Entro in perform_all_fits")
    for index, col in enumerate(df.columns):
        # Parametri iniziali per il fit
        initial_guess = [1000, 150]
        # Step 3: Eseguire il fitting
        # y_data = df['0_1057']
        # y_data = df[col]
        filtered_df = df_to_consider_for_fit[
            (df_to_consider_for_fit.index >= selected_range[0]) & (df_to_consider_for_fit.index <= selected_range[1])]
        params, covariance = curve_fit(exponential, filtered_df.index, filtered_df[col], p0=initial_guess)

        # Stampare i parametri del fit
        a, tau = params
        # print(f"Parametri del fit: a={a}, b={tau}")
        # all_found_taus.append(tau / 24.0)
        all_found_taus[index] = (tau / 24.0)
        # print(f"Fit ch{index}: a = {a}, tau= {tau}\n")
        # Step 4: Calcolare i valori previsti dal modello
        # y_fit = exponential(df.index, *params)


def generate_meas_intervals(meas_to_gen, min_value, max_value, min_distance):
    print("Entro in generate_meas_intervals, ne faccio", meas_to_gen)
    numbers = []
    while len(numbers) < meas_to_gen:
        num = random.uniform(min_value, max_value)
        # Check if the number is sufficiently far from all others
        if all(abs(num - existing) >= min_distance for existing in numbers):
            numbers.append(num)
    return numbers


# Caricamento dati con gestione errori
try:
    df = pd.read_csv('rate_Lu_IFO_soglie13_Overweekend2401_senzacorrezione.csv', parse_dates=['tempo_h'])
except FileNotFoundError:
    print("Errore: File CSV non trovato.")
    df = pd.DataFrame()
except Exception as e:
    print(f"Errore nel caricamento CSV: {e}")
    df = pd.DataFrame()

# rate_Lu_IFO_soglie13_Overweekend2401_senzacorrezione
# Leggi il file CSV
# df = pd.read_csv('rate_Lu_IFO_soglie13_Overweekend2401_senzacorrezione.csv', parse_dates=['tempo_h'])
# df = pd.read_csv('Lu_IFO_2001_senzacorrezione.csv', parse_dates=['tempo_h'])
# df['tempo_h'] = pd.to_datetime(df['tempo_h'], unit="h")
df = df.set_index('tempo_h')
df['time_delta'] = (df.index - df.index[0]).total_seconds() / 60 / 60
df = df.set_index('time_delta')
# df = df.drop("tempo_h", axis=1)
df_to_consider_for_fit = df

total_duration_min = len(df) * 100 / 60

numberOfMeas = 10
measDurationInMin = 30
minDistanceBetweenMeasInMin = 180

generated_measurements = []

all_found_t12 = [0] * 10
real_t12 = 6.6

# generated_measurements = generate_meas_intervals(numberOfMeas, total_duration_min, minDistanceBetweenMeasInMin)

print(
    f"Caricato dataset: è lungo {len(df)} misure, quindi {total_duration_min:.0f}min, {len(df) / 3600 * 100:.1f}h")

# for meas in generated_measurements:
#    print(f"Misura generata: {meas:.1f} min")
# Inizializzazione dell'app Dash
app = dash.Dash(__name__)
server = app.server


# Funzione per generare un grafico scatter con Seaborn
def generate_scatterplot(x_range=None):
    # print("entro in generate_scatterplot", x_range, selected_fraction)
    plt.figure(figsize=(6, 4))
    # if selected_fraction:
    #     df_to_show = df.sample(frac=selected_fraction / 100, random_state=42)
    # else:
    #     df_to_show = df
    for col in df:
        sns.scatterplot(data=df_to_consider_for_fit, x=df_to_consider_for_fit.index, y=df_to_consider_for_fit[col],
                        alpha=0.4, label=col)

    if x_range:
        # plt.xlim(x_range)
        plt.axvline(x=x_range[0], color="salmon", linestyle="--", linewidth=1, label="_nolegend_")
        plt.axvline(x=x_range[1], color="salmon", linestyle="--", linewidth=1, label="_nolegend_")

    for meas in generated_measurements:
        # plt.axvline(x=meas/60, color="blue", linestyle="--", linewidth=1, label="Vertical Line")
        plt.axvspan((meas - measDurationInMin) / 60, (meas + measDurationInMin) / 60, color='limegreen', alpha=0.3,
                    label="_nolegend_")

    plt.ylabel("Raw Rate [CPS]")
    plt.xlabel("Time [h]")
    plt.legend(title="Channels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # plt.legend(title="Sensors", loc="upper left")

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
                                html.H4("Selezione Range", id="range-values"),
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
                        html.Div(
                            children=[
                                html.Div([
                                    html.H4("Selezione Numero misure", id="meas-number"),
                                    dcc.Checklist(
                                        id='only-meas-checkbox',
                                        options=[{'label': 'Solo misure', 'value': 'filter'}],
                                        value=[],
                                        inline=True
                                    )
                                ], style={'display': 'flex', 'align-items': 'center', 'gap': '10px'}),
                                dcc.Slider(
                                    id="meas-number-slider",
                                    min=1,
                                    max=20,
                                    step=1,
                                    value=5,
                                    marks={i: str(i) for i in range(0, 51, 2)},
                                ),
                                dcc.Slider(
                                    id="meas-duration-slider",
                                    min=1,
                                    max=60,
                                    step=5,
                                    value=30,
                                    marks={i: str(i) for i in range(0, 61, 2)},
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


@app.callback(
    [
        Output("meas-number", "children"),
        Output("range-values", "children"),
        Output("fraction-title", "children")
    ],
    [
        Input("meas-number-slider", "value"),
        Input("meas-duration-slider", "value"),
        Input("range-slider", "value"),
        Input("fraction-slider", "value")
    ]
)
def update_texts(meas_value, meas_duration_value, range_value, fraction_value):
    return (
        f"Selezione Numero misure: {meas_value} da {meas_duration_value} min",
        f"Range selezionato: {range_value} h",
        f"Selezione Frazione: {fraction_value}"
    )


# Callback per aggiornare il dataset quando viene mosso qualche controllo
@app.callback(
    Output("scatterplot", "src"),
    Output("all-taus", "figure"),
    Output("all-taus-diff", "figure"),
    Input("range-slider", "value"),
    Input("fraction-slider", "value"),
    Input("meas-number-slider", "value"),
    Input("meas-duration-slider", "value"),
    Input("only-meas-checkbox", "value"),
)
def update_plots(selected_range, selected_fraction, meas_to_generate, meas_duration, use_only_meas):
    print(f"Entro in update_plots chiamato da {ctx.triggered_id}\n\n")
    # triggered_id = ctx.triggered_id
    # print("Trigger è stato", triggered_id)
    global df_to_consider_for_fit  # Cosi recupero la globale e non ne creo una nuova
    global generated_measurements
    global measDurationInMin
    measDurationInMin = meas_duration
    """Genera le misure sperimentali e trovane gli indici estremi nel df"""
    generated_measurements = generate_meas_intervals(meas_to_generate, min(selected_range) * 60
                                                     , max(selected_range) * 60, minDistanceBetweenMeasInMin)
    # for meas in generated_measurements:
    #    print(f"Misura generata: {meas:.1f} min")
    indices_couples = []
    for generate_meas in generated_measurements:
        nearest_index_start = df.index[abs((df.index - (generate_meas - measDurationInMin / 2) / 60.0)).argmin()]
        nearest_index_end = df.index[abs((df.index - (generate_meas + measDurationInMin / 2) / 60.0)).argmin()]
        row_number_start = df.index.get_loc(nearest_index_start)
        row_number_end = df.index.get_loc(nearest_index_end)
        indices_couples.append([row_number_start, row_number_end])
        #    nearest_row = df.loc[nearest_index]

        print(
            f"\tCerco {generate_meas:.0f}min ({generate_meas / 60.0:.2f}h) (+- {measDurationInMin / 2}min), indice piu vicino {nearest_index_start:.2f}-{nearest_index_end:.2f}, numero riga: {row_number_start}-{row_number_end}\n\t\tLista:{len(indices_couples)}-{indices_couples}")

    """Filtra il dataset da usare"""
    if use_only_meas:
        # Seleziona e concatena le righe
        df_to_consider_for_fit = pd.concat([df.iloc[start:end] for start, end in indices_couples])
        print(f"Prendo solo {len(df_to_consider_for_fit)} misure da 100s ({meas_to_generate} da {measDurationInMin})")
    else:
        if selected_fraction != 100:
            print("Resamplo con frazione ", selected_fraction)
            df_to_consider_for_fit = df.sample(frac=selected_fraction / 100)
        else:
            print("Prendo tutto il df")
            df_to_consider_for_fit = df

    """Genera il grafico generale dei dati"""
    img_data = generate_scatterplot(x_range=selected_range)
    buf = io.BytesIO()
    img_data.save(buf, format="PNG")
    buf.seek(0)
    encoded_img_data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    """Genera il grafico dei t1/2"""
    global all_found_t12
    perform_all_fits(all_found_t12, selected_range)

    fig_t12 = px.bar(
        x=range(10),
        y=all_found_t12,
        labels={"x": "Channel", "y": "T1/2 [d]"},
        title="T1/2 fittati",
    )
    fig_t12.update_layout(
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
    fig_t12.update_layout(
        yaxis=dict(range=[min(all_found_t12) * 0.95, max(all_found_t12) * 1.05]),  # Range fisso per l'asse Y
    )

    """Genera il grafico dei t1/2 DIFF"""
    fig_t12_diff = px.bar(
        x=range(10),
        y=[(x - real_t12) * 100 for x in all_found_t12 if x != 0],
        labels={"x": "Channel", "y": "Diff T1/2 [%]"},
        title="Errori % su T1/2",
    )

    fig_t12_diff.update_layout(
        yaxis=dict(range=[-100, 100]),  # Range fisso per l'asse Y
    )

    fig_t12_diff.update_traces(marker_color="coral")

    print("Risultato fit t1/2:\n", all_found_t12)
    print("Risultato fit err:\n", [(x - real_t12) * 100 for x in all_found_t12 if x != 0])

    return encoded_img_data, fig_t12, fig_t12_diff


# Esecuzione dell'app
if __name__ == "__main__":
    app.run_server(debug=True)
