import dash
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.MINTY]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
