import time
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output
import dash_table as dt
import plotly.graph_objs as go
import pandas as pd

from SubTreesClasses import read_data_to_df, annotate_data
from app import app
import plotly.express as px

# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='page-content')
# ])
from const.Constants import SPACE, SEMI_COLON, EMPTY_STR

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H2(" ", className="text-center"), className="text-center")
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card(children=[html.H4(children='Разметка медицинского корпуса',
                                           className="text-center"),
                                   dbc.Button(["Разметка корпуса анамнезов",
                                               dbc.Badge("Пример", color="light", className="ml-1")],
                                              color="primary",
                                              className="mt-1",
                                              id="loading-button"),
                                   # dbc.Button("Разметка своего корпуса",
                                   #            color="primary",
                                   #            className="mt-1"),
                                   #
                                   ],
                         body=True, color="success", outline=True)
                , width=6, className="mb-4")
        ], justify="center", align="center", className="text-center"),

        # dbc.Row([
        #     html.A("Разметка использует информацию, заложенную в синтаксической структуре предложения."),
        #     html.A("Для получения этой информации используются морфологический и синтаксический анализаторы."),
        #     html.A("Разметить можно как уже предварительно разобранный анализаторами корпус, так и корпус исходных предложений,"
        #            + " который приложение в таком случае разберет."),
        #     html.A(
        #         "Разработанный алгоритм группирует похожие по контексту фрагменты предложений и размечает получившиеся группы" +
        #         " с использованием данных, доступных в медицинских базах знаний."),
        #     html.A("Результат разметки наглядно представлен и автоматически скачивается."),
        # ], justify="center", align="center", className="text-center"),

        dbc.Row([
            dbc.Col(html.H2(" ", className="text-center"), className="text-center")
        ]),
        html.H5("Время: "),
        html.Div(id='read-time'),
        # html.Br(),
        html.Div(id='w2v-time'),
        # html.Br(),
        html.Div(id='const-time'),
        # html.Br(),
        html.Div(id='algo-time'),
        # html.Br(),
        html.Div(id='label-time'),
        html.Br(),
        html.H5("Общая информация: "),
        html.Div(id='num_sentences'),
        # html.Br(),
        html.Div(id='num_classes_labeled'),
        # html.Br(),
        html.Div(id='num_labels'),

        dbc.Row([
            dbc.Col(html.H5("30 наиболее частых меток", className="text-center"), className="text-center")
        ]),
        html.Div([
            dcc.Graph(id='graph', style={"height": "80vh", "width": "100vh"}),
            #     dt.DataTable(id='data-table', columns=[
            #     {'name': 'Метка', 'id': 'title'},
            #     {'name': 'Количество размеченных классов', 'id': 'score'}
            # ])
            dbc.Row([
                dbc.Col(html.H5("Статистика по размеру полученных классов", className="text-center"),
                        className="text-center")
            ]),
            dbc.Row([
                html.Div([
                    dcc.Graph(
                        id='graph_1', style={"height": "55vh", "width": "55vh"}
                    ), ], className='res_hist '),
                html.Div([
                    dcc.Graph(
                        id='graph_2', style={"height": "55vh", "width": "55vh"}
                    ), ], className='res_hist'),
            ])
        ]),
        # dash_table.DataTable(id='graph_3', )
        dbc.Row([
            dbc.Col(html.H5("Аннотированные классы", className="text-center"),
                    className="text-center")
        ]),
        html.Div([
            dcc.Graph(id='graph_3', style={"height": "100vh", "width": "120vh"}),
        ], className='table')
    ])  # TODO: поменять текст, чтобы было красивее!
])

@app.callback([Output("graph", "figure"), Output("graph_1", "figure"), Output("graph_2", "figure"), Output("graph_3", "figure"), Output('read-time', 'children'), Output('w2v-time', 'children'), Output('const-time', 'children'),
               Output('algo-time', 'children'), Output('label-time', 'children'), Output('num_sentences', 'children'), Output('num_classes_labeled', 'children'),
               Output('num_labels', 'children')],
              Input("loading-button", "n_clicks"))
def input_triggers_spinner(n_clicks):
    if n_clicks:
        overall_time, label_classes_sorted, results_dict, num_sentences, num_classes_labeled, result_sent_dict, class_id_labels_full = annotate_data()
        label_classes_simple = {k: len(v) for k, v in label_classes_sorted.items()}
        top_30 = list(label_classes_simple.items())[-30:]
        labels = [item[0] for item in top_30]
        counts = [item[1] for item in top_30]

        # hist for displaying number of words in a single repeat of a class
        remapped_dict_1 = {k: len(v[0].split(SPACE)) for k, v in results_dict.items()}
        repeat_len_dict = {}
        for k, v in remapped_dict_1.items():
            if v not in repeat_len_dict.keys():
                repeat_len_dict[v] = 1
            else:
                repeat_len_dict[v] += 1

        # hist for displaying number of repeats in a class
        remapped_dict_2 = {k: len(v) for k, v in results_dict.items()}
        class_len_dict = {}
        for k, v in remapped_dict_2.items():
            if v not in class_len_dict.keys():
                class_len_dict[v] = 1
            else:
                class_len_dict[v] += 1

        # columns = [
        #     {'name': k.capitalize(), 'id': k}
        #     for k in label_classes_sorted.keys()
        # ]
        # figure = go.Figure(data=[go.Bar(x=classes_num, text=label, name=label) for label, classes_num in label_classes_sorted.items()])
        # layout = go.Layout(title={'text':'30 наиболее частых меток', 'x':0.5, 'xanchor': 'center'}, bargap=0.30)
        # layout_1 = go.Layout(title={'text': 'Число слов в повторе', 'x': 0.5, 'xanchor': 'center'}, bargap=0.30)
        # layout_2 = go.Layout(title={'text': 'Число повторов в классе', 'x': 0.5, 'xanchor': 'center'}, bargap=0.30)

        layout = go.Layout(bargap=0.30)
        layout_1 = go.Layout(bargap=0.30)
        layout_2 = go.Layout(bargap=0.30)
        # num_sentences = 0 # TEST
        # num_classes_labeled = 0 # TEST
        # label_classes_simple = {} # TEST
        # overall_time = [0,0,0,0,0] # TEST
        # figure = go.Figure({'data': []}) # TEST
        # figure_1 = go.Figure({'data': []})  # TEST
        # figure_2 = go.Figure({'data': []})  # TEST
        figure = go.Figure(data=[go.Bar(x=counts,
                                        y=labels,
                                        orientation='h',
                                        marker=dict(color='#20c997', line=dict(color='#49A249', width=3))),],
                                        # marker=dict(color='#2672F7')),],
                           layout=layout)
        figure_1 = go.Figure(data=[go.Bar(x=list(repeat_len_dict.values()),
                                        y=list(repeat_len_dict.keys()),
                                        orientation='h',
                                        marker=dict(color='#20c997', line=dict(color='#49A249', width=3))),],
                           layout=layout_1)
        figure_2 = go.Figure(data=[go.Bar(x=list(class_len_dict.values()),
                                        y=list(class_len_dict.keys()),
                                        orientation='h',
                                        marker=dict(color='#20c997', line=dict(color='#49A249', width=3))),],
                           layout=layout_2)

        # figure_1.update_xaxes(
        #     tickangle=90,
        #     title_text="Month",
        #     title_font={"size": 20},
        #     title_standoff=25)
        figure.update_xaxes(title_text="Число классов")
        figure.update_yaxes(title_text="Метка")

        figure_1.update_xaxes(title_text="Число классов")
        figure_1.update_yaxes(title_text="Число слов в повторе")

        figure_2.update_xaxes(title_text="Число классов")
        figure_2.update_yaxes(title_text="Число повторов в классе")

        column_classes = []
        column_labels = []
        column_sents = []
        column_repeat = []
        results_dict_labeled = {k: v for k, v in results_dict.items() if k in class_id_labels_full.keys()} # only labeled classes
        count = 1
        for class_id, repeat_list in results_dict_labeled.items():
            sents = result_sent_dict[class_id]
            for index, repeat in enumerate(repeat_list):
                if index == 0:
                    column_classes.append(str(count))
                    count += 1
                else:
                    column_classes.append(EMPTY_STR)
                if index == 0:
                    column_labels.append(SEMI_COLON.join(list(class_id_labels_full[class_id])))
                else:
                    column_labels.append(EMPTY_STR)
                column_sents.append(sents[index])
                column_repeat.append(repeat)

        figure = go.Figure(data=[go.Bar(x=counts,
                                        y=labels,
                                        orientation='h',
                                        marker=dict(color='#20c997', line=dict(color='#49A249', width=3))), ],
                           layout=go.Layout(
                               title=go.layout.Title(text="30 наиболее частых меток")
                           ))

        figure_3 = go.Figure(data=[go.Table(columnwidth=[5, 10, 6, 15], header=dict(values=['Класс', 'Метка', 'Имя файла', 'Повтор']),
                                       cells=dict(values=[column_classes, column_labels, column_sents, column_repeat], fill_color='#BEECB5'))
                              ])

            # ,title_standoff=25)
        return [figure, figure_1, figure_2, figure_3,
                u' Чтение: {} с.'.format("%.2f" % overall_time[0]),
                u' Построение векторного пространства: {} с.'.format("%.2f" % overall_time[1]),
                u' Построение общего синтаксического дерева: {} с.'.format("%.2f" % overall_time[2]),
                u' Алгоритм поиска схожих поддеревьев: {} с.'.format("%.2f" % overall_time[3]),
                u' Постобработка и разметка: {} с.'.format("%.2f" % overall_time[4]),
                u' Число предложений в корпусе: {}'.format(num_sentences),
                u' Число размеченных классов: {}'.format(num_classes_labeled),
                u' Число уникальных меток: {}'.format(len(label_classes_simple.keys())),]
    return [go.Figure({'data': []}), go.Figure({'data': []}), go.Figure({'data': []}), go.Figure({'data': []}), '', '', '', '', '', '', '', '']


if __name__ == '__main__':
    app.run_server(debug=True)
