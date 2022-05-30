# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 01:16:24 2020

@author: joser
"""


from datetime import datetime, timedelta
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
from controls import LOANTYPE, PROGRAMS, LENDERTYPE, EATYPE, TARGET1, TARGET2, verdicts
import pathlib
import math

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data-source").resolve()
MODEL_PATH = PATH.joinpath("saved-models").resolve()

#styling/css
external_stylesheets = ['https://dash-gallery.plotly.host/dash-oil-and-gas/assets/s1.css','https://dash-gallery.plotly.host/dash-oil-and-gas/assets/styles.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#read dataset
ea_clusters = pd.read_csv(DATA_PATH.joinpath('ea_clusters_spyder.csv'),low_memory=False)
lender_clusters = pd.read_csv(DATA_PATH.joinpath('lender_clusters_spyder.csv'),low_memory=False)
dataset = pd.read_csv(DATA_PATH.joinpath('indonesia-sovereign-debt-dataset-1998-above.csv'),low_memory=False)
bb = pd.read_csv(DATA_PATH.joinpath('blue-book-2020-2024.csv'),encoding='unicode_escape',sep=';',low_memory=False)

#datatable_parameter
PAGE_SIZE = 200

#import model 
#model 1 classification numerous drawing limit amendment
model_1_vc = pickle.load(open(MODEL_PATH.joinpath("finalized_model.sav"), "rb"))
model_1_rf = pickle.load(open(MODEL_PATH.joinpath("finalized_model_a.sav"), "rb"))
model_1_knn = pickle.load(open(MODEL_PATH.joinpath("finalized_model_b.sav"), "rb"))
model_1_logreg = pickle.load(open(MODEL_PATH.joinpath("finalized_model_c.sav"), "rb"))
#model 2 multiclass classification disbursement ratio percentage
model_2_vc = pickle.load(open(MODEL_PATH.joinpath("finalized_model2.sav"), "rb"))
model_2_rf = pickle.load(open(MODEL_PATH.joinpath("finalized_model2_a.sav"), "rb"))
model_2_knn = pickle.load(open(MODEL_PATH.joinpath("finalized_model2_b.sav"), "rb"))
model_2_logreg = pickle.load(open(MODEL_PATH.joinpath("finalized_model2_c.sav"), "rb"))

def predictBlueBook_disbursement_category(a,b,c,d,e,f,g,h):
    
    eac = a    
    lc = b
    
    var1 = math.log10(c)
    var2 = 365*d
    var3 = (h==1)
    var4 = (h==2)
    var5 = (e==1)
    var6 = (e==2)
    var7 = (e==3)
    var22 = (f==1)
    var23 = (f==2)
    var8 = (eac==3)
    var9 = (eac==2)
    var10 = (eac==1)
    var11 = (eac==4)
    var12 = (g==1)
    var13 = (g==2)
    var14 = (g==3)
    var15 = (g==4)
    var16 = (g==5)
    var17 = (g==6)
    var18 = (lc==1)
    var19 = (lc==4)
    var20 = (lc==3)
    var21 = (lc==2)    
    
    pred1 = model_2_vc.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
    
    return pred1

def predictBlueBook_numerous_drawing_limit_amendment(a,b,c,d,e,f,g,h):
    
    eac = a    
    lc = b
    
    var1 = math.log10(c)
    var2 = 365*d
    var3 = (h==1)
    var4 = (h==2)
    var5 = (e==1)
    var6 = (e==2)
    var7 = (e==3)
    var22 = (f==1)
    var23 = (f==2)
    var8 = (eac==3)
    var9 = (eac==2)
    var10 = (eac==1)
    var11 = (eac==4)
    var12 = (g==1)
    var13 = (g==2)
    var14 = (g==3)
    var15 = (g==4)
    var16 = (g==5)
    var17 = (g==6)
    var18 = (lc==1)
    var19 = (lc==4)
    var20 = (lc==3)
    var21 = (lc==2)    
    
    pred2 = model_1_vc.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
    
    return str(pred2)

def leMillion(number):
    return number*1000

bb['nilai_pinjaman'] = bb.apply(lambda row: leMillion(row.loan_val_mil_usd), axis = 1)
bb['disb_predict'] =  bb.apply(lambda row: predictBlueBook_disbursement_category(row.ea_cluster,1,row.loan_val_mil_usd*1000000,row.ap,row.project_type_code,row.ea_type_code,4,row.loan_type_code), axis = 1)
bb['dl_amend_predict'] =  bb.apply(lambda row: predictBlueBook_numerous_drawing_limit_amendment(row.ea_cluster,1,row.loan_val_mil_usd*1000000,row.ap,row.project_type_code,row.ea_type_code,4,row.loan_type_code), axis = 1)
BlueBook_dataset2 = bb[['Project_Name','ea','ea_type','loan_type','nilai_pinjaman','project_type','disb_predict','dl_amend_predict']]
BlueBook_dataset2[' index'] = range(1, len(BlueBook_dataset2) + 1)

#lender summary
lender_sum_df = dataset.groupby('LENDER_NAME', as_index=False).agg({"USD_AMT_ORI":"sum"}).sort_values('USD_AMT_ORI',ascending = False).head(5)
lender_count_df = dataset.groupby('LENDER_NAME', as_index=False).agg({"USD_AMT_ORI":"count"}).sort_values('USD_AMT_ORI',ascending = False).head(5)

#ea summary
ea_sum_df = dataset.groupby('EXECUTING_AGENCY_NAME', as_index=False).agg({"USD_AMT_ORI":"sum"}).sort_values('USD_AMT_ORI',ascending = False).head(5)
ea_count_df = dataset.groupby('EXECUTING_AGENCY_NAME', as_index=False).agg({"USD_AMT_ORI":"count"}).sort_values('USD_AMT_ORI',ascending = False).head(5)
summ_dfs = dataset.groupby('LENDER_TIPE', as_index=False).agg({"USD_AMT_ORI":"count"})
#markdown
markdown_text_ea = '''

Pembagian *executing agency* ke dalam 4 kluster berdasarkan hasil analisis menggunakan *elbow method*. 
Pada kluster 1, kluster 2, dan kluster 4, akan didapati executing agency dengan frekuensi perolehn pinjaman pemerintah RI di bawah 300 kali. 

**Kluster 1** diperuntukkan bagi executing agency yang pernah menerima pinjaman pemerintah kurang dari sama dengan 10 kali dengan nilai komitmen pinjaman rata-rata di bawah USD 700 juta. 

Sedangkan **kluster 2** diperuntukkan bagi executing agency dengan frekuensi perolehan pinjaman pemerintah yang juga rendah yakni di bawah 300 kali, namun dengan besaran nilai komitmen pinjaman rata-rata yang sangat tinggi yakni minimal USD 45 Miliar. Kementerian Keuangan, Polri, Kementerian Kesehatan, serta beberapa BUMN seperti PT PGN dan PT Pertamina masuk dalam kluster ini. 

Berikutnya **kluster 3**  adalah executing agency dengan frekuensi perolehan pinjaman pemerintah RI yang sangat sering, yaitu di atas 400 komitmen pinjaman. Ada 4 executing agency yang masuk kluster ini. Yakni Kementerian PUPR, PT PLN (Persero), Kementerian Perhubungan, serta Kementerian Pertahanan.

Sementara itu, **kluster 4** terdiri atas executing agency dengan frekuensi perolehan pinjaman pemerintah yang rendah serta rata-rata nila komitmen pinjaman yang juga rendah, yakni BPPT, BP Batam, serta Perpustakaan Nasional. 
'''

markdown_text_lender = '''

Pembagian *lender* ke dalam 4 kluster berdasarkan hasil analisis menggunakan *elbow method*. 
Pada kluster 4, kluster 2, dan kluster 3, akan didapati lender dengan frekuensi pemberian pinjaman kepada pemerintah RI di bawah 200 kali. 

**Kluster 1**   adalah lender dengan frekuensi pinjaman kepada pemerintah RI yang sangat sering, yaitu di atas 200 komitmen pinjaman. Ada 6 lender yang masuk kluster ini. Yakni IBRD, JICA (Jepang), ADB, KfW (Jerman), OECF, dan BFCE (Perancis). 

Sedangkan **kluster 2** diperuntukkan bagi lender dengan nilai komitmen pinjaman rata-rata di bawah USD 6 miliar. 

Berikutnya **kluster 3** terdiri atas 4 lender dengan frekuensi pemberian pinjaman yang rendah serta rata-rata nila komitmen pinjaman yang juga rendah, antara lain Credit Agricole dan Asian Infrastructure Investment Bank (AIIB). 

Sementara itu **kluster 4** diperuntukkan bagi lender dengan frekuensi pinjaman kepada pemerintah RI yang juga rendah yakni di bawah 90 kali, namun dengan besaran nilai komitmen pinjaman rata-rata yang sangat tinggi yakni minimal USD 6,4 Miliar. Bank BUMN dalam negeri yakni Bank Mandiri, BNI, dan BRI masuk dalam kluster ini. 

'''
######dash html components stays here

app.layout = html.Div(children=[
    
        html.Div(#start of header div
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("kemenkeu-logo.png"),
                            id="logo-image",
                            style={
                                "height": "100px",
                                "width": "auto",
                            },
                        ),
                        html.Div(
                            [
                                html.H3(
                                    "Republic of Indonesia Sovereign Debt",
                                    style={"margin-bottom": "0px", "font-weight":"bold"},
                                ),
                                html.H5(
                                    "Performance Predictive Analytics", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="twelve columns",
                    id="title",
                )
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
            ),#end of header div
        
            dcc.Tabs([##start of tab
                
                ###first tab starts
                dcc.Tab(label='Informasi Umum', children=[
                    html.Div([
                        html.Div([
                            html.H5("Skema Prediksi", style={"font-weight":"bold"}),
                            html.P("Model ini dikembangkan dengan tujuan memprediksi kinerja pinjaman pemerintah RI, utamanya terhadap potensi keterlambatan realisasi proyek yang dibiayai melalui pinjaman."),
                            html.P("Potensi keterlambatan tersebut diproksikan melalui dua hal, yakni banyaknya amandemen atas batas waktu pencairan pinjaman serta tingkat pencairan pinjaman yang rendah."),
                            html.P("Model prediksi mengunakan algoritma ensemble voting classifier yang disusun dengan 3 (tiga) algoritma dasar yakni random forest, logistic regression, dan kNN clasifier.")
                            ],
                            className="pretty_container three columns"
                            ),   
                        html.Div([
                            html.Img(
                                src=app.get_asset_url("model-scheme.png"),
                                id="scheme-image",
                                style={
                                    "height": "auto",
                                    "width": "90%",
                                    "margin-bottom": "25px",
                                    },
                                ),
                            html.P("Model prediktif ini dikembangkan dengan menggunakan 5 (lima) jenis prediktor utama yang terdiri atas karakteristik lender, karakteristik executing agency, nilai pinjaman, availability period, serta jenis proyek yang dibiayai melalui pinjaman tersebut.")
                            ],
                            id="predictiveDescription",
                            className="pretty_container nine columns",
                            style={
                                "text-align":"center"
                                }
                            )
                        ],
                        className="row flex-display"
                        ),
                    ]),
                ###first tab ends
                
                ###second tab starts
                dcc.Tab(label='Deskripsi Data', children=[
                    html.Div([ #start row div for viz-1
                        html.Div([
                            html.H5("Pinjaman Pemerintah RI per Jenis Executing Agency", style={"font-weight":"bold"}),
                            html.Div([
                                dcc.RadioItems(
                                    id='value_type',
                                    options=[{'label': i, 'value': i} for i in ['Jumlah', 'Nilai']],
                                    value='Jumlah',
                                    labelStyle={'display': 'inline-block'}
                                    ),
                                dcc.Graph(id='per-ea-comparison-graph',
                                          style={'height':600}),
                                ],style={'width': '98%'}),
                            dcc.Graph(id="summ_ea"),
                            ],className="twelve columns"),
                        ],className="pretty_container row flex-display"), #end row div
                    html.Div([ #start row div for viz-2
                        html.Div([
                            html.H5("Pinjaman Pemerintah RI per Jenis Lender", style={"font-weight":"bold"}),
                            html.Div([
                                dcc.RadioItems(
                                    id='value_type2',
                                    options=[{'label': i, 'value': i} for i in ['Jumlah', 'Nilai']],
                                    value='Jumlah',
                                    labelStyle={'display': 'inline-block'}
                                    ),
                                dcc.Graph(id='per-lender-comparison-graph',
                                          style={'height':600}),
                                ],style={'width': '98%'}),
                            dcc.Graph(id="summ_lender"),
                            ],className="twelve columns"),
                        ],className="pretty_container row flex-display"), #end row div
                    html.Div([ #start row div for secodn viz with two graphs
                        html.Div([
                            dcc.Graph(id="summ_target1"),
                            ],className="pretty_container six columns"),
                        html.Div([
                            dcc.Graph(id="summ_target2"),
                            ],className="pretty_container six columns")
                        ],className="row flex-display"), #end row div
                    html.Div([ #start row div for viz-4
                        html.Div([
                            dcc.Graph(id="summ_target1_top"),
                            dcc.Graph(id="summ_target2_top"),
                            ],className="pretty_container twelve columns")
                        ],className="row flex-display"), #end row div
                    ]),
                ###second tab ends
                
                ###third tab starts
                dcc.Tab(label='Analisis Kluster', children=[
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H5('Cluster Lender', style={"font-weight":"bold"})
                                ]),
                            dcc.Graph(
                                id='lender-clusters',
                                figure={
                                    'data': [
                                        go.Scatter(
                                            x=lender_clusters[lender_clusters['Kluster'] == i]['LOAN_COUNT'],
                                            y=lender_clusters[lender_clusters['Kluster'] == i]['LOG_LOAN_AVERAGE_AMT_USD'],
                                            text=lender_clusters[lender_clusters['Kluster'] == i]['LENDER_NAME'],
                                            mode='markers',
                                            opacity=0.8,
                                            marker={
                                                'size': 14,
                                                'line': {'width': 0.5, 'color': 'white'}
                                                },
                                            name=i
                                            ) for i in lender_clusters.Kluster.unique()
                                        ],
                                    'layout': go.Layout(
                                        xaxis={'title': 'Count of Loan Commitments'},
                                        yaxis={'title': 'Average Loan Amount (Log)'},
                                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                        legend={'x': 0, 'y': 1},
                                        hovermode='closest'
                                        )
                                    }
                                ),
                            html.P(),
                            dcc.Markdown(children=markdown_text_lender)          
                            ],
                            className="pretty_container six columns"
                            ),
                        html.Div([
                             html.Div([
                                 html.H5('Cluster Executing Agency',style={"font-weight":"bold"})
                                 ]),
                             dcc.Graph(
                                 id='ea-clusters',
                                 figure={
                                     'data': [
                                         go.Scatter(
                                             x=ea_clusters[ea_clusters['Kluster'] == i]['LOAN_COUNT'],
                                             y=ea_clusters[ea_clusters['Kluster'] == i]['LOG_LOAN_AVG_AMT'],
                                             text=ea_clusters[ea_clusters['Kluster'] == i]['EA_NAME'],
                                             mode='markers',
                                             opacity=0.8,
                                             marker={
                                                 'size': 14,
                                                 'line': {'width': 0.5, 'color': 'white'}
                                                 },
                                             name=i
                                             ) for i in ea_clusters.Kluster.unique()
                                         ],
                                     'layout': go.Layout(
                                         xaxis={'title': 'Count of Loan Commitments'},
                                         yaxis={'title': 'Average Loan Amount (Log)'},
                                         margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                         legend={'x': 0, 'y': 1},
                                         hovermode='closest'
                                         )
                                     }
                                 ),
                             html.P(),
                             dcc.Markdown(children=markdown_text_ea)       
                             ],
                            className="pretty_container six columns"
                            ),
                        ],
                        className="row flex-display"
                        ),
                    ]),
                ###third tab ends
                
                ###fourth tab starts
                dcc.Tab(label='Analisis Prediktif', children=[
                    html.Div([ #start of row div for all predictors
                        html.Div([#start of 6 column div for lender and ea characteristics
                            html.Div([#start of lender characteristic predictors
                                html.H5("Karakteristik Lender", style={"font-weight":"bold"}),
                                html.H6("Nama Lender:", className="dcc_control"),
                                dcc.Dropdown(
                                    id="lender-cluster-dropdown",
                                    options=[{'label': i, 'value': i} for i in lender_clusters['LENDER_NAME']],
                                    value='JAPAN INTERNATIONAL COOPERATION AGENCY',
                                    className="dcc_control",
                                    ),
                                html.P(id="lc-title",className="dcc_control"),
                                html.H6("Jenis Lender", className="control_label"),
                                dcc.RadioItems(
                                    id="lender_type_selector",
                                    options=[
                                        {"label": "Kreditor Swasta Asing (KSA)", "value": 1},
                                        {"label": "Lembaga Penjamin Kredit Ekspor (LPKE)", "value": 2},
                                        {"label": "Lembaga Multilateral", "value": 3},
                                        {"label": "Negara (Bilateral)", "value": 4},
                                        {"label": "Bank BUMN Dalam Negeri", "value": 5},
                                        {"label": "Bank Swasta Dalam Negeri", "value": 6}
                                        ],
                                    value=3,
                                    labelStyle={"display": "inline-block"},
                                    className="dcc_control",
                                    ),
                                ],
                                className="pretty_container twelve columns"
                                ),#end of lender characteristics predictors
                            html.Div([#start of executing agency characteristics predictors
                                html.H5("Karakteristik Executing Agency", style={"font-weight":"bold"}),
                                html.H6("Nama Executing Agency:", className="dcc_control"),
                                dcc.Dropdown(
                                    id="ea-cluster-dropdown",
                                    options=[{'label': i, 'value': i} for i in ea_clusters['EA_NAME']],
                                    value='KEMENTERIAN KESEHATAN',
                                    className="dcc_control",
                                    ),
                                html.P(id="eac-title",className="dcc_control"),
                                html.H6("Jenis Executing Agency", className="control_label"),
                                dcc.RadioItems(
                                    id="ea_type_selector",
                                    options=[
                                        {"label": "Kementerian/Lembaga", "value": 1},
                                        {"label": "Badan Usaha Milik Negara", "value": 2},
                                        {"label": "Pemerintah Daerah", "value": 3}
                                        ],
                                    value=2,
                                    labelStyle={"display": "inline-block"},
                                    className="dcc_control",
                                    ),
                                ],
                                className="pretty_container twelve columns"
                                ),#end of executing agency characteristics predictors
                            ],
                            className="six columns"
                            ),#end of 6 columns div for ea and lender characteristics predictors
                        html.Div([#start of loan characteristics predictors
                            html.H5("Karakteristik Pinjaman", style={"font-weight":"bold"}),
                            html.H6("Jenis Pinjaman", className="control_label"),
                            dcc.RadioItems(
                                id="loan_type_selector",
                                options=[
                                    {"label": "Pinjaman Luar Negeri", "value": 1},
                                    {"label": "Pinjaman Dalam Negeri", "value": 2}
                                    ],
                                value=1,
                                labelStyle={"display": "inline-block"},
                                className="dcc_control",
                                ),
                            html.H6(id="ap-title", className="control_label"),
                            html.P("Merupakan jangka waktu antara tanggal efektif pinjaman sampai dengan closing date.", className="control_label"),
                            dcc.Slider(
                                id='ap-slider',
                                min=0,
                                max=20,
                                value=10,
                                marks={str(x): str(x) for x in (0,5,10,15,20)},
                                className="dcc_control"
                                ),
                            html.H6("Nilai Pinjaman (dalam USD):", className="control_label"),
                            dcc.Input(
                                id="val-slider",
                                type="number",
                                value=1000000000000,
                                placeholder="Nilai Pinjaman (USD)",
                                ),
                            html.P(id="val-title", className="control_label"),                    
                            html.H6("Jenis Proyek", className="control_label"),
                            dcc.RadioItems(
                                id="project_type_selector",
                                options=[
                                    {"label": "Pengadaan Barang Publik", "value": 1},
                                    {"label": "Program Pemerintah", "value": 2},
                                    {"label": "Pembangunan Infrastruktur", "value": 3}
                                    ],
                                value=2,
                                labelStyle={"display": "inline-block"},
                                className="dcc_control",
                                ),
                            ],
                            className="pretty_container six columns"
                            ),#end of container and six columns div for loan characteristics predictors
                        ],
                        className="row flex-display"
                        ),#end of row div for all predictors
                    #prediction result is here
                    html.Div([#start of result row prediction
                        html.Div([#start of prediction column
                            html.H5('Hasil Prediksi Model 1', style={"font-weight":"bold"}),
                            html.P("Model prediksi 1 merupakan multiclass classification untuk memprediksi keterlambatan realisasi proyek yang dibiayai melalui pinjaman, dengan menggunakan proksi tingkat pencairan pinjaman (disbursement category). Model 1 menghasilkan 3 (tiga) jenis prediksi yakni 1. pinjaman dengan pencairan di bawah 50% dari nilai komitmen (DISBURSEMENT UNDER 50%), 2. pinjaman dengan pencairan antara 50-90% dari nilai komitmen (DISBURSEMENT UNDER 90%), 3. pinjaman dengan pencairan lebih dari 90% nilai komitmen (FULLY DISBURSED)"),
                            html.H1(id="pred-1", style={"font-weight":"bold"}),
                            html.H6('Kesimpulan Prediksi:', style={"font-weight":"bold"}),
                            html.P(id='verdict1')
                            ],
                            className="pretty_container six columns"
                            ),#end of prediction column
                        html.Div([#start of prediction column
                            html.H5('Hasil Prediksi Model 2', style={"font-weight":"bold"}),
                            html.P("Model prediksi 2 merupakan binary classification untuk memprediksi keterlambatan  realisasi proyek yang dibiayai melalui pinjaman, dengan menggunakan proksi banyaknya amandemen atas batas waktu pencairan pinjaman (date drawing limit). Model 2 menghasilkan 2 (dua) jenis prediksi yakni True dan False, yang menyatakan apakah pinjaman tersebut akan mengalami lebih dari dua kali amandemen atas date drawing limit."),
                            html.H1(id="pred-2", style={"font-weight":"bold"}),
                            html.H6('Kesimpulan Prediksi:', style={"font-weight":"bold"}),
                            html.P(id='verdict2')
                            ],
                            className="pretty_container six columns"
                            ),#end of prediction column
                        ],className="row flex-display"
                        ),#end of row prediction
                    #individual model prediction            
                    html.Div([#start of row div for individual model prediction
                        html.Div([
                            html.H6('Model 1 - Random Forest', style={"font-weight":"bold"}),
                            html.P(),
                            html.H5(id="pred-1-a")
                            ],
                            className="pretty_container two columns"
                            ),
                        html.Div([
                            html.H6('Model 1 - Log. Regression', style={"font-weight":"bold"}),
                            html.P(),
                            html.H5(id="pred-1-b")
                            ],
                            className="pretty_container two columns"
                            ),
                        html.Div([
                            html.H6('Model 1 - k-NN Classifier', style={"font-weight":"bold"}),
                            html.P(),
                            html.H5(id="pred-1-c")
                            ],
                            className="pretty_container two columns"
                            ),
                        html.Div([
                            html.H6('Model 2 - Random Forest', style={"font-weight":"bold"}),
                            html.P(),
                            html.H5(id="pred-2-a")
                            ],
                            className="pretty_container two columns"
                            ),
                        html.Div([
                            html.H6('Model 2 - Log. Regression', style={"font-weight":"bold"}),
                            html.P(),
                            html.H5(id="pred-2-b")
                            ],
                            className="pretty_container two columns"
                            ),
                        html.Div([
                            html.H6('Model 2 - k-NN Classifier', style={"font-weight":"bold"}),
                            html.P(),
                            html.H5(id="pred-2-c")
                            ],
                            className="pretty_container two columns"
                            ),
                        ],
                        className="row flex-display"
                        ),#end of row div
                    ]),#tab close
                ###fourth tab ends
                
                ###fifth tab starts
                dcc.Tab(label='DRPLN-JM 2020-2024', children=[#tab starts
                    html.Div([
                        html.Div([
                            html.H6('Prediksi Rencana Pinjaman Luar Negeri dalam DRPLN-JM 2020-2024', style={"font-weight":"bold"}),
                            html.P("Berikut ini adalah hasil analisis prediktif atas rencana pinjaman luar negeri Pemerintah RI sebagaimana termuat dalam DRPLN-JM (Blue Book) Tahun 2020-2024. Karena dalam Blue Book belum terdapat lender atas masing-masing rencana kegiatan beserta perkiraan jangka waktu pelaksanaan kegiatan tersebut, maka ditetapkan terlebih dahulu perkiraan lender, serta diasumsikan bahwa availability period pinjaman selama 5 tahun."),
                            html.H5("Karakteristik Lender", style={"font-weight":"bold"}),
                            html.H6("Nama Lender:", className="dcc_control"),
                            dcc.Dropdown(
                                id="lender-cluster-dropdown-bb",
                                options=[{'label': i, 'value': i} for i in lender_clusters['LENDER_NAME']],
                                value='ISLAMIC DEVELOPMENT BANK',
                                className="dcc_control",
                                ),
                            html.P(id="lc-title-bb",className="dcc_control"),
                            html.H6("Jenis Lender", className="control_label"),
                            dcc.RadioItems(
                                id="lender_type_selector_bb",
                                options=[
                                    {"label": "Kreditor Swasta Asing (KSA)", "value": 1},
                                    {"label": "Lembaga Penjamin Kredit Ekspor (LPKE)", "value": 2},
                                    {"label": "Lembaga Multilateral", "value": 3},
                                    {"label": "Negara (Bilateral)", "value": 4},
                                    {"label": "Bank BUMN Dalam Negeri", "value": 5},
                                    {"label": "Bank Swasta Dalam Negeri", "value": 6}
                                    ],
                                value=3,
                                labelStyle={"display": "inline-block"},
                                className="dcc_control",
                                ),
                            dash_table.DataTable(
                                id='bluebook-table-paging-and-sorting',
                                columns=[
                                    {'name': i, 'id': i, 'deletable': True} for i in BlueBook_dataset2.columns
                                    ],
                                page_current=0,
                                page_size=PAGE_SIZE,
                                page_action='custom',
                                sort_action='custom',
                                sort_mode='single',
                                sort_by=[],
                                style_cell={
                                    'whiteSpace': 'normal',
                                    'height': 'auto',
                                    },
                                style_cell_conditional=[
                                    {
                                        'if': {'column_id': i},
                                        'textAlign': 'left'
                                        } for i in ['Project_Name', 'ea','loan_type','project_type','disb_predict','dl_amend_predict']
                                    ],
                                )                            
                            ],className="pretty_container twelve columns")
                        ],className="row flex-display")
                    ]),#tab close
                ###fifth tab ends
                
                ]),#end of overall dcc.tabs
            
            html.Div([#start of footer div
                html.P("© 2020 - Inspektorat Jenderal Kementerian Keuangan", style={"font-weight":"bold"})
                ],className="pretty_container", style={'text-align':'center'}
                )#end of footer div
    
    ]) #end of app.layout

######end of all dash html components

@app.callback(
        Output("ap-title", "children"),
        [Input("ap-slider", "value")])

def apTitle(value):
    return 'Availability Period ({} Tahun)'.format(value)

@app.callback(
        Output("val-title", "children"),
        [Input("val-slider", "value")])

def valTitle(value):
    return 'Nilai Pinjaman USD {}'.format(value)

@app.callback(
        Output("lc-title", "children"),
        [Input("lender-cluster-dropdown", "value")])

def lcTitle(value):
    cluster = lender_clusters['Kluster'].values[lender_clusters['LENDER_NAME']==value] 
    return 'Cluster Lender: {} '.format(cluster)

@app.callback(
        Output("lc-title-bb", "children"),
        [Input("lender-cluster-dropdown-bb", "value")])

def lcTitlebb(value):
    cluster = lender_clusters['Kluster'].values[lender_clusters['LENDER_NAME']==value] 
    return 'Cluster Lender: {} '.format(cluster)

@app.callback(
        Output("eac-title", "children"),
        [Input("ea-cluster-dropdown", "value")])

def eacTitle(value):
    cluster = ea_clusters['Kluster'].values[ea_clusters['EA_NAME']==value] 
    return 'Cluster EA: {} '.format(cluster)

@app.callback(
        Output("pred-1", "children"),
        Output("pred-2", "children"),
        Output("pred-1-a", "children"),
        Output("pred-1-b", "children"),
        Output("pred-1-c", "children"),
        Output("pred-2-a", "children"),
        Output("pred-2-b", "children"),
        Output("pred-2-c", "children"),
        Output("verdict1", "children"),
        Output("verdict2", "children"),
        [Input("ea-cluster-dropdown", "value"),
         Input("lender-cluster-dropdown", "value"),
         Input("val-slider", "value"),
         Input("ap-slider", "value"),
         Input("project_type_selector", "value"),
         Input("ea_type_selector", "value"),
         Input("lender_type_selector", "value"),
         Input("loan_type_selector", "value")
         ])

def predictLoan(a,b,c,d,e,f,g,h):
    
    eacluster = ea_clusters['cluster_pred'].values[ea_clusters['EA_NAME']==a]
    eac = eacluster+1
    
    lendcluster = lender_clusters['cluster_pred'].values[lender_clusters['LENDER_NAME']==b]
    lc = lendcluster+1
    
    var1 = math.log10(c)
    var2 = 365*d
    var3 = (h==1)
    var4 = (h==2)
    var5 = (e==1)
    var6 = (e==2)
    var7 = (e==3)
    var22 = (f==1)
    var23 = (f==2)
    var8 = (eac==3)
    var9 = (eac==2)
    var10 = (eac==1)
    var11 = (eac==4)
    var12 = (g==1)
    var13 = (g==2)
    var14 = (g==3)
    var15 = (g==4)
    var16 = (g==5)
    var17 = (g==6)
    var18 = (lc==1)
    var19 = (lc==4)
    var20 = (lc==3)
    var21 = (lc==2)    
    
    pred1 = model_2_vc.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
    pred2 = model_1_vc.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
    pred1a = model_2_rf.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
    pred1b = model_2_logreg.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
    pred1c = model_2_knn.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
    pred2a = model_1_rf.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
    pred2b = model_1_logreg.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
    pred2c = model_1_knn.predict([[var1, var2, var3, var4, var5, var6, var7, var22, var23, var8, var9, var10, var11, var12, var13, var14, var15, var16, var17, var18, var19, var20, var21]])
        
    recommendation1 = "{} yang berasal dari lender {} (Cluster {}) senilai USD {} dengan jangka waktu (availability period) {} tahun, untuk keperluan pembiayaan {} yang dilaksanakan oleh executing agency {} (Cluster {}) {}".format(LOANTYPE[h-1],LENDERTYPE[g-1],lc,c,d,PROGRAMS[e-1],EATYPE[f-1],eac,verdict1(pred1))
    
    recommendation2 = "{} yang berasal dari lender {} (Cluster {}) senilai USD {} dengan jangka waktu (availability period) {} tahun, untuk keperluan pembiayaan {} yang dilaksanakan oleh executing agency {} (Cluster {}) {}".format(LOANTYPE[h-1],LENDERTYPE[g-1],lc,c,d,PROGRAMS[e-1],EATYPE[f-1],eac,verdict2(str(pred2)))
    
    return pred1, str(pred2), pred1a, pred1b, pred1c, str(pred2a), str(pred2b), str(pred2c), recommendation1, recommendation2

def verdict1(pred1):
    if(pred1=="DISBURSEMENT UNDER 50%"):
        return "diprediksikan akan memiliki tingkat pencairan/realisasi yang rendah yakni di bawah 50% dari nilai komitmen"
    elif(pred1=="DISBURSEMENT UNDER 90%"):
        return "diprediksikan akan memiliki tingkat pencairan/realisasi antara 50-90% dari nilai komitmen"
    elif(pred1=="FULLY DISBURSED"):
        return "diprediksikan akan memiliki tingkat pencairan/kinerja realisasi yang baik, yakni di atas 90% nilai komitmen pinjaman dapat direalisasikan"
    else:
        return ""

def verdict2(pred2):
    if(pred2=="[True]"):
        return "diprediksikan akan mengalami lebih dari dua kali amandemen date drawing limit"
    elif(pred2=="[False]"):
        return "diprediksikan tidak akan mengalami lebih dari dua kali amandemen date drawing limit"
    else:
        return ""

@app.callback(
    Output('per-ea-comparison-graph', 'figure'),
    Output('summ_ea', 'figure'),
    [Input('value_type', 'value')])
def update_figure_comparison(type):
    if(type=="Nilai"):
        summ_df = dataset.groupby('EA_TIPE', as_index=False).agg({"USD_AMT_ORI":"sum"})
        fig = px.pie(summ_df, values='USD_AMT_ORI', names='EA_TIPE', color_discrete_sequence=px.colors.sequential.RdBu)
        #chart title and transition
        fig.update_layout(transition_duration=500)
        fig2 = go.Figure(go.Bar(y=ea_sum_df['EXECUTING_AGENCY_NAME'], x=ea_sum_df['USD_AMT_ORI'], xaxis='x2', yaxis='y2', orientation='h'))
        fig2.layout.update({'title': '5 Executing Agency dengan Total Nilai Pinjaman Terbesar 1998-2020'})
        return fig,fig2
    else:
        summ_df = dataset.groupby('EA_TIPE', as_index=False).agg({"USD_AMT_ORI":"count"})
        fig = px.pie(summ_df, values='USD_AMT_ORI', names='EA_TIPE', color_discrete_sequence=px.colors.sequential.RdBu)
        #chart title and transition
        fig.update_layout(transition_duration=500)
        fig2 = go.Figure(go.Bar(y=ea_count_df['EXECUTING_AGENCY_NAME'], x=ea_count_df['USD_AMT_ORI'], xaxis='x2', yaxis='y2', orientation='h'))
        fig2.layout.update({'title': '5 Executing Agency dengan Total Jumlah Pinjaman Terbanyak 1998-2020'})
        return fig, fig2
    
@app.callback(
    Output('per-lender-comparison-graph', 'figure'),
    Output('summ_lender', 'figure'),
    Output('summ_target1', 'figure'),
    Output('summ_target2', 'figure'),
    Output('summ_target1_top', 'figure'),
    Output('summ_target2_top', 'figure'),
    [Input('value_type2', 'value')])
def update_figure_comparison2(type):
    pre_summ_target3 = dataset[dataset.NUMEROUS_DRAWING_LIMIT_AMENDMENT==True]
    pre_summ_target4 = dataset[dataset.DISBURSEMENT_CATEGORY=="DISBURSEMENT UNDER 50%"]
    summ_target3 = pre_summ_target3.groupby('EXECUTING_AGENCY_NAME', as_index=False).agg({"USD_AMT_ORI":"count"}).sort_values('USD_AMT_ORI',ascending = False).head(5)
    summ_target4 = pre_summ_target4.groupby('EXECUTING_AGENCY_NAME', as_index=False).agg({"DISBURSEMENT_CATEGORY":"count"}).sort_values('DISBURSEMENT_CATEGORY',ascending = False).head(5)
    summ_target1 = dataset.groupby('NUMEROUS_DRAWING_LIMIT_AMENDMENT', as_index=False).agg({"USD_AMT_ORI":"count"})
    summ_target2 = dataset.groupby('DISBURSEMENT_CATEGORY', as_index=False).agg({"USD_AMT_ORI":"count"})
    fig3 = go.Figure(go.Bar(x=summ_target1['NUMEROUS_DRAWING_LIMIT_AMENDMENT'], y=summ_target1['USD_AMT_ORI'], xaxis='x2', yaxis='y2', marker=dict(color='#20c997')))
    fig3.layout.update({'title': 'Amandemen Drawing Limit 1998-2020'})
    fig4 = go.Figure(go.Bar(x=summ_target2['DISBURSEMENT_CATEGORY'], y=summ_target2['USD_AMT_ORI'], xaxis='x2', yaxis='y2', marker=dict(color='#dc3545')))
    fig4.layout.update({'title': 'Persentase Pencairan Pinjaman) 1998-2020'})
    fig5 = go.Figure(go.Bar(x=summ_target3['EXECUTING_AGENCY_NAME'], y=summ_target3['USD_AMT_ORI'], xaxis='x2', yaxis='y2', marker=dict(color='#ffc107')))
    fig5.layout.update({'title': '5 Executing Agency Terbanyak Amandemen Drawing Limit Pinjaman'})
    fig6 = go.Figure(go.Bar(x=summ_target4['EXECUTING_AGENCY_NAME'], y=summ_target4['DISBURSEMENT_CATEGORY'], xaxis='x2', yaxis='y2', marker=dict(color='#6f42c1')))
    fig6.layout.update({'title': '5 Executing Agency Terbanyak Pinjaman Pencairan Rendah'})
    if(type=="Nilai"):
        summ_df = dataset.groupby('LENDER_TIPE', as_index=False).agg({"USD_AMT_ORI":"sum"})
        fig = px.pie(summ_df, values='USD_AMT_ORI', names='LENDER_TIPE', color_discrete_sequence=px.colors.sequential.RdBu)
        #chart title and transition
        fig.update_layout(transition_duration=500)
            #legend setting
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            ))
        fig2 = go.Figure(go.Bar(y=lender_sum_df['LENDER_NAME'], x=lender_sum_df['USD_AMT_ORI'], xaxis='x2', yaxis='y2', orientation='h'))
        fig2.layout.update({'title': '5 Lender dengan Total Nilai Pinjaman Terbesar 1998-2020'})
        
        return fig, fig2, fig3, fig4, fig5, fig6
    else:
        summ_df = dataset.groupby('LENDER_TIPE', as_index=False).agg({"USD_AMT_ORI":"count"})
        fig = px.pie(summ_df, values='USD_AMT_ORI', names='LENDER_TIPE', color_discrete_sequence=px.colors.sequential.RdBu)
        #chart title and transition
        fig.update_layout(transition_duration=500)
            #legend setting
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            ))
        fig2 = go.Figure(go.Bar(y=lender_count_df['LENDER_NAME'], x=lender_count_df['USD_AMT_ORI'], xaxis='x2', yaxis='y2', orientation='h'))
        fig2.layout.update({'title': '5 Lender dengan Total Jumlah Pinjaman Terbanyak 1998-2020'})
        return fig, fig2, fig3, fig4, fig5, fig6
    
@app.callback(
    Output('bluebook-table-paging-and-sorting', 'data'),
    [Input('bluebook-table-paging-and-sorting', 'page_current'),
     Input('bluebook-table-paging-and-sorting', 'page_size'),
     Input('bluebook-table-paging-and-sorting', 'sort_by'),
     Input('lender-cluster-dropdown-bb', 'value'),
     Input('lender_type_selector_bb', 'value')])
def update_table_bluebook(page_current, page_size, sort_by,lender_cluster_bb,lender_type_bb):
    #preprocess blue book dataset
    lendcluster = lender_clusters['cluster_pred'].values[lender_clusters['LENDER_NAME']==lender_cluster_bb]
    lc = lendcluster+1
    bb['nilai_pinjaman'] = bb.apply(lambda row: leMillion(row.loan_val_mil_usd), axis = 1)
    bb['disb_predict'] =  bb.apply(lambda row: predictBlueBook_disbursement_category(row.ea_cluster,lc,row.loan_val_mil_usd*1000,row.ap,row.project_type_code,row.ea_type_code,lender_type_bb,row.loan_type_code), axis = 1)
    bb['dl_amend_predict'] =  bb.apply(lambda row: predictBlueBook_numerous_drawing_limit_amendment(row.ea_cluster,lc,row.loan_val_mil_usd*1000,row.ap,row.project_type_code,row.ea_type_code,lender_type_bb,row.loan_type_code), axis = 1)
    #bb['disb_predict'] =  predictBlueBook_disbursement_category(row.ea_cluster,lc,bb['loan_val_mil_usd']*1000000,bb['ap'],bb['project_type_code'],bb['ea_type_code'],lender_type_bb,bb['loan_type_code'])
    #bb['dl_amend_predict'] =  bb.apply(lambda row: predictBlueBook_numerous_drawing_limit_amendment(row.ea_cluster,lc,row.loan_val_mil_usd*1000000,row.ap,row.project_type_code,row.ea_type_code,lender_type_bb,row.loan_type_code), axis = 1)
    BlueBook_dataset = bb[['Project_Name','ea','ea_type','loan_type','nilai_pinjaman','project_type','disb_predict','dl_amend_predict']]
    BlueBook_dataset[' index'] = range(1, len(BlueBook_dataset) + 1)
    if len(sort_by):
        dff = BlueBook_dataset.sort_values(
            sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc',
            inplace=False
        )
    else:
        # No sort is applied
        dff = BlueBook_dataset

    return dff.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)