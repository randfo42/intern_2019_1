import dash

app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
