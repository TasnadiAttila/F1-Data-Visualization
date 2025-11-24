import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import fastf1

# Cache beállítása - engedélyezve, a hiányzó munkamenetek letöltése és cache-elése folytatható
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# Globális cache az adatoknak
DATA_CACHE = {}
SCHEDULE_CACHE = {}

def get_data(year):
    year = int(year)
    if year in DATA_CACHE:
        return DATA_CACHE[year]
    
    filename = f'{year}_full_season_data.xlsx'
    df_stints = pd.DataFrame()
    df_results = pd.DataFrame()
    
    if os.path.exists(filename):
        print(f"Adatok betöltése innen: {filename}")
        try:
            df_stints = pd.read_excel(filename, sheet_name='Stints_Summary')
            if not df_stints.empty:
                df_stints['Year'] = year
        except Exception as e:
            print(f"Hiba az Excel (Stints) betöltésekor ({year}): {e}")

        try:
            df_results = pd.read_excel(filename, sheet_name='Results_Full')
            if not df_results.empty:
                df_results['Year'] = year
        except Exception as e:
            print(f"Hiba az Excel (Results) betöltésekor ({year}): {e}")
    else:
        print(f"Nincs adatfájl ehhez az évhez: {year}. Futtasd az 'extract_data.py'-t!")

    DATA_CACHE[year] = (df_stints, df_results)
    return df_stints, df_results

def get_schedule_data(year):
    year = int(year)
    if year in SCHEDULE_CACHE:
        return SCHEDULE_CACHE[year]
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        SCHEDULE_CACHE[year] = schedule
        return schedule
    except Exception as e:
        print(f"Hiba a naptár letöltésekor ({year}): {e}")
        return pd.DataFrame()

# Segédfüggvény opciók generálásához
def get_options(years):
    if not isinstance(years, list):
        years = [years]
    
    all_race_options = []
    drivers_set = set()
    teams_set = set()
    driver_labels = {}
    
    for year in sorted(years):
        print(f"Processing options for year: {year}")
        # Races
        schedule = get_schedule_data(year)
        if not schedule.empty:
            for _, row in schedule.iterrows():
                label = f"{year} Round {row['RoundNumber']}: {row['EventName']}"
                value = f"{year}_{row['RoundNumber']}"
                all_race_options.append({'label': label, 'value': value})
        else:
            print(f"Warning: Schedule is empty for {year}")
        
        # Drivers & Teams
        _, df_results = get_data(year)
        if not df_results.empty:
            try:
                # Teams
                team_col = 'TeamName' if 'TeamName' in df_results.columns else ('Team' if 'Team' in df_results.columns else None)
                if team_col:
                    teams = df_results[team_col].dropna().unique().astype(str)
                    teams_set.update(teams)

                # Drivers
                if 'Driver' in df_results.columns and 'Abbreviation' in df_results.columns:
                    for _, row in df_results[['Driver', 'Abbreviation']].drop_duplicates().iterrows():
                        abbr = row['Abbreviation']
                        name = row['Driver']
                        drivers_set.add(abbr)
                        driver_labels[abbr] = f"{name} ({abbr})"
                elif 'Abbreviation' in df_results.columns:
                    drivers = df_results['Abbreviation'].unique().astype(str)
                    drivers_set.update(drivers)
                    for d in drivers:
                        if d not in driver_labels:
                            driver_labels[d] = d
            except Exception as e:
                print(f"Hiba a versenyzők/csapatok listájának összeállításakor ({year}): {e}")
        else:
            print(f"Warning: Results data is empty for {year}")

    # Sort and format
    team_options = [{'label': t, 'value': t} for t in sorted(list(teams_set))]
    driver_options = [{'label': driver_labels.get(d, d), 'value': d} for d in sorted(list(drivers_set))]
            
    return all_race_options, driver_options, team_options# Kezdeti adatok (2022)
DEFAULT_YEAR = 2022
init_race_options, init_driver_options, init_team_options = get_options([DEFAULT_YEAR])

# Alkalmazás inicializálása
app = dash.Dash(__name__)

app.layout = html.Div([
    # Header / Control Bar
    html.Div([
        html.H1("F1 Szezon Elemző", style={'margin': '0', 'fontSize': '24px', 'fontWeight': '600'}),
        html.Div([
            html.Div([
                html.Label("Év:", style={'marginRight': '10px', 'color': '#fff'}),
                dcc.Dropdown(
                    id='year-selector',
                    options=[{'label': str(y), 'value': y} for y in range(2021, 2025)],
                    value=[DEFAULT_YEAR],
                    multi=True,
                    clearable=False,
                    style={'width': '200px', 'color': '#000', 'marginRight': '20px'}
                ),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                html.Label("Felső sorban:", style={'marginRight': '10px', 'color': '#fff'}),
                dcc.Dropdown(
                    id='layout-selector',
                    options=[
                        {'label': 'Gumistratégiák', 'value': 'tire'},
                        {'label': 'Kvalifikáció', 'value': 'qual'},
                        {'label': 'Verseny', 'value': 'race'},
                        {'label': 'Bajnokság', 'value': 'champ'},
                        {'label': 'Pozíció Változás', 'value': 'gain'},
                        {'label': 'Köridők', 'value': 'lap'}
                    ],
                    value=['tire', 'qual'],
                    multi=True,
                    clearable=False,
                    style={'width': '300px', 'color': '#000'}
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '20px'}),
            html.Button("Nézet váltása", id='btn-layout-toggle', n_clicks=0, className='btn-modern')
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={
        'display': 'flex', 
        'justifyContent': 'space-between', 
        'alignItems': 'center', 
        'padding': '20px 40px', 
        'backgroundColor': '#252526', 
        'borderBottom': '1px solid #333',
        'marginBottom': '30px'
    }),

    html.Div(id='graphs-container', style={'display': 'flex', 'flexDirection': 'column', 'padding': '0 20px'}, children=[
        
        # Left Container (Tire Strategy)
        html.Div(id='left-container', className='card', children=[
            html.H3("Gumistratégiák", style={'marginTop': '0'}),
            html.Div([
                html.Label("Válassz futamot:"),
                dcc.Dropdown(
                    id='race-dropdown',
                    options=init_race_options,
                    value=init_race_options[0]['value'] if init_race_options else None,
                    clearable=False
                ),
            ], style={'width': '100%', 'marginBottom': '20px'}),
            
            dcc.Loading(
                id="loading-1",
                type="default",
                children=dcc.Graph(id='tire-graph', style={'height': '800px'})
            ),
        ], style={'flex': '1', 'minWidth': '400px', 'marginRight': '0'}),

        # Right Container (Qualifying Results)
        html.Div(id='right-container', className='card', children=[
            html.H3("Kvalifikációs Eredmények", style={'marginTop': '0'}),
            html.Div([
                html.Div([
                    html.Label("Válassz csapatot:"),
                    dcc.Dropdown(
                        id='qual-team-dropdown',
                        options=init_team_options,
                        multi=True,
                        placeholder="Válassz csapatot..."
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1002'}),
                html.Div([
                    html.Label("Válassz versenyzőket:"),
                    dcc.Dropdown(
                        id='qual-driver-dropdown',
                        options=init_driver_options,
                        multi=True,
                        placeholder="Válassz versenyzőket..."
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1001'}),
                html.Div([
                    html.Label("Válassz futamokat:"),
                    dcc.Dropdown(
                        id='qual-race-dropdown',
                        options=init_race_options,
                        multi=True,
                        placeholder="Válassz futamokat...",
                        value=[r['value'] for r in init_race_options] if init_race_options else []
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1000'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'width': '100%', 'marginBottom': '60px'}),
            
            html.Div([
                dcc.Loading(
                    id="loading-2",
                    children=dcc.Graph(
                        id='qual-graph', 
                        style={'height': '800px', 'width': '100%', 'display': 'block'},
                        responsive=True
                    ),
                    type="default",
                    style={'width': '100%'}
                )
            ], style={'width': '100%', 'flex': '1'})
        ], style={'flex': '1', 'minWidth': '400px', 'display': 'flex', 'flexDirection': 'column'}),

        # Race Results Container
        html.Div(id='race-results-container', className='card', children=[
            html.H3("Verseny Eredmények", style={'marginTop': '0'}),
            html.Div([
                html.Div([
                    html.Label("Válassz csapatot:"),
                    dcc.Dropdown(
                        id='race-team-dropdown',
                        options=init_team_options,
                        multi=True,
                        placeholder="Válassz csapatot..."
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1002'}),
                html.Div([
                    html.Label("Válassz versenyzőket:"),
                    dcc.Dropdown(
                        id='race-driver-dropdown',
                        options=init_driver_options,
                        multi=True,
                        placeholder="Válassz versenyzőket..."
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1001'}),
                html.Div([
                    html.Label("Válassz futamokat:"),
                    dcc.Dropdown(
                        id='race-race-dropdown',
                        options=init_race_options,
                        multi=True,
                        placeholder="Válassz futamokat...",
                        value=[r['value'] for r in init_race_options] if init_race_options else []
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1000'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'width': '100%', 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Loading(
                    id="loading-3",
                    children=dcc.Graph(
                        id='race-graph', 
                        style={'height': '800px', 'width': '100%', 'display': 'block'},
                        responsive=True
                    ),
                    type="default",
                    style={'width': '100%'}
                )
            ], style={'width': '100%', 'flex': '1'})
        ], style={'flex': '1', 'minWidth': '400px', 'display': 'flex', 'flexDirection': 'column'}),

        # Championship Standings Container
        html.Div(id='championship-container', className='card', children=[
            html.H3("Világbajnoki Pontverseny", style={'marginTop': '0'}),
            html.Div([
                html.Div([
                    html.Label("Válassz csapatot:"),
                    dcc.Dropdown(
                        id='champ-team-dropdown',
                        options=init_team_options,
                        multi=True,
                        placeholder="Válassz csapatot..."
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1002'}),
                html.Div([
                    html.Label("Válassz versenyzőket:"),
                    dcc.Dropdown(
                        id='champ-driver-dropdown',
                        options=init_driver_options,
                        multi=True,
                        placeholder="Válassz versenyzőket..."
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1001'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'width': '100%', 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Loading(
                    id="loading-4",
                    children=dcc.Graph(
                        id='champ-graph', 
                        style={'height': '800px', 'width': '100%', 'display': 'block'},
                        responsive=True
                    ),
                    type="default",
                    style={'width': '100%'}
                )
            ], style={'width': '100%', 'flex': '1'})
        ], style={'flex': '1', 'minWidth': '400px', 'display': 'flex', 'flexDirection': 'column'}),

        # Position Gain/Loss Container
        html.Div(id='gain-loss-container', className='card', children=[
            html.H3("Pozíció Nyereség/Veszteség", style={'marginTop': '0'}),
            html.Div([
                html.Div([
                    html.Label("Válassz csapatot:"),
                    dcc.Dropdown(
                        id='gain-team-dropdown',
                        options=init_team_options,
                        multi=True,
                        placeholder="Válassz csapatot..."
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1002'}),
                html.Div([
                    html.Label("Válassz versenyzőket:"),
                    dcc.Dropdown(
                        id='gain-driver-dropdown',
                        options=init_driver_options,
                        multi=True,
                        placeholder="Válassz versenyzőket..."
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1001'}),
                html.Div([
                    html.Label("Válassz futamokat:"),
                    dcc.Dropdown(
                        id='gain-race-dropdown',
                        options=init_race_options,
                        multi=True,
                        placeholder="Válassz futamokat...",
                        value=[r['value'] for r in init_race_options] if init_race_options else []
                    )
                ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1000'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'width': '100%', 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Loading(
                    id="loading-5",
                    children=dcc.Graph(
                        id='gain-graph', 
                        style={'height': '800px', 'width': '100%', 'display': 'block'},
                        responsive=True
                    ),
                    type="default",
                    style={'width': '100%'}
                )
            ], style={'width': '100%', 'flex': '1'})
        ], style={'flex': '1', 'minWidth': '400px', 'display': 'flex', 'flexDirection': 'column'}),

        # Lap Time Distribution Container
        html.Div(id='lap-dist-container', className='card', children=[
            html.H3("Köridők Eloszlása", style={'marginTop': '0'}),
            html.Div([
                html.Div([
                    html.Label("Válassz futamot:"),
                    dcc.Dropdown(
                        id='lap-race-dropdown',
                        options=init_race_options,
                        value=init_race_options[0]['value'] if init_race_options else None,
                        clearable=False
                    ),
                ], style={'width': '100%', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Label("Válassz csapatot:"),
                        dcc.Dropdown(
                            id='lap-team-dropdown',
                            options=init_team_options,
                            multi=True,
                            placeholder="Válassz csapatot..."
                        )
                    ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1002'}),
                    html.Div([
                        html.Label("Válassz versenyzőket:"),
                        dcc.Dropdown(
                            id='lap-driver-dropdown',
                            options=init_driver_options,
                            multi=True,
                            placeholder="Válassz versenyzőket..."
                        )
                    ], style={'width': 'calc(50% - 10px)', 'marginBottom': '10px', 'position': 'relative', 'zIndex': '1001'}),
                ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px', 'width': '100%'}),
            ], style={'width': '100%', 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Loading(
                    id="loading-6",
                    children=dcc.Graph(
                        id='lap-graph', 
                        style={'height': '800px', 'width': '100%', 'display': 'block'},
                        responsive=True
                    ),
                    type="default",
                    style={'width': '100%'}
                )
            ], style={'width': '100%', 'flex': '1'})
        ], style={'flex': '1', 'minWidth': '400px', 'display': 'flex', 'flexDirection': 'column'})
    ])
])

@app.callback(
    [Output('graphs-container', 'style'),
     Output('left-container', 'style'),
     Output('right-container', 'style'),
     Output('race-results-container', 'style'),
     Output('championship-container', 'style'),
     Output('gain-loss-container', 'style'),
     Output('lap-dist-container', 'style')],
    [Input('btn-layout-toggle', 'n_clicks'),
     Input('layout-selector', 'value')]
)
def toggle_layout(n_clicks, selected_top):
    base_card_style = {'flex': '1', 'minWidth': '400px', 'display': 'flex', 'flexDirection': 'column'}
    
    # Default styles (Vertical)
    container_style = {'display': 'flex', 'flexDirection': 'column', 'padding': '0 20px', 'gap': '20px'}
    left_style = {**base_card_style, 'width': '100%', 'order': 0}
    right_style = {**base_card_style, 'width': '100%', 'order': 0}
    race_style = {**base_card_style, 'width': '100%', 'order': 0}
    champ_style = {**base_card_style, 'width': '100%', 'order': 0}
    gain_style = {**base_card_style, 'width': '100%', 'order': 0}
    lap_style = {**base_card_style, 'width': '100%', 'order': 0}

    if n_clicks % 2 != 0:
        # Horizontal (Grid) Mode
        container_style = {'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap', 'alignItems': 'stretch', 'padding': '0 20px', 'gap': '20px'}
        
        # Helper to determine style based on selection
        def get_grid_style(key):
            if key in selected_top:
                return {**base_card_style, 'width': 'calc(50% - 10px)', 'flex': '0 0 calc(50% - 10px)', 'order': 1}
            else:
                return {**base_card_style, 'width': '100%', 'flex': '0 0 100%', 'order': 2}

        left_style = get_grid_style('tire')
        right_style = get_grid_style('qual')
        race_style = get_grid_style('race')
        champ_style = get_grid_style('champ')
        gain_style = get_grid_style('gain')
        lap_style = get_grid_style('lap')

    return container_style, left_style, right_style, race_style, champ_style, gain_style, lap_style

@app.callback(
    Output('tire-graph', 'figure'),
    [Input('race-dropdown', 'value'),
     Input('year-selector', 'value')]
)
def update_graph(race_value, years):
    if not race_value:
        return go.Figure().update_layout(title="Válassz egy futamot!")
    
    try:
        year, round_number = map(int, str(race_value).split('_'))
    except ValueError:
        return go.Figure().update_layout(title="Érvénytelen futam kiválasztás.")

    df_all_data, df_results = get_data(year)
    try:
        # Ha van előfeldolgozott adat, használjuk azt
        if not df_all_data.empty:
            df_stints = df_all_data[df_all_data['RoundNumber'] == round_number].copy()
            
            if df_stints.empty:
                return go.Figure().update_layout(title="Nincs elérhető adat ehhez a futamhoz a CSV-ben.")
            
            event_name = df_stints['EventName'].iloc[0]
            
            # Eredmények betöltése az abbreviációkhoz (ha elérhető a Results_Full lap)
            abbr_map = {}
            if not df_results.empty:
                try:
                    df_r = df_results[df_results['RoundNumber'] == round_number]
                    for _, rrow in df_r.iterrows():
                        abbr = None
                        if 'Abbreviation' in df_r.columns and pd.notna(rrow.get('Abbreviation')):
                            abbr = str(rrow['Abbreviation']).strip()
                        if not abbr:
                            continue
                        # Map several possible keys to the abbreviation
                        # BroadcastName (display name)
                        if 'BroadcastName' in df_r.columns and pd.notna(rrow.get('BroadcastName')):
                            abbr_map[str(rrow['BroadcastName']).strip()] = abbr
                        # Driver (may hold full name)
                        if 'Driver' in df_r.columns and pd.notna(rrow.get('Driver')):
                            abbr_map[str(rrow['Driver']).strip()] = abbr
                        # DriverNumber (map numeric start numbers)
                        if 'DriverNumber' in df_r.columns and pd.notna(rrow.get('DriverNumber')):
                            try:
                                num = int(rrow['DriverNumber'])
                                abbr_map[str(num)] = abbr
                            except Exception:
                                pass
                        # Forename/Surname combine if available
                        if 'Forename' in df_r.columns and 'Surname' in df_r.columns and pd.notna(rrow.get('Surname')):
                            full = f"{rrow.get('Forename','')} {rrow.get('Surname','')}".strip()
                            if full:
                                abbr_map[full] = abbr
                        # Also map abbreviation to itself
                        abbr_map[abbr] = abbr
                except Exception:
                    abbr_map = {}

            # Konvertáló függvény az abbreviációk előállításához
            def to_abbr(name):
                s = str(name).strip()
                if not s:
                    return s
                # If already a 3-letter uppercase code, keep it
                if len(s) == 3 and s.isupper():
                    return s

                # Direct map lookup
                if s in abbr_map:
                    return abbr_map[s]

                # If s is numeric (driver number), try matching
                if s.isdigit() and s in abbr_map:
                    return abbr_map[s]
                try:
                    n = int(float(s))
                    if str(n) in abbr_map:
                        return abbr_map[str(n)]
                except Exception:
                    pass

                # Case-insensitive partial matches
                sl = s.lower()
                for k, v in abbr_map.items():
                    if not k:
                        continue
                    kl = str(k).lower()
                    if kl in sl or sl in kl:
                        return v

                # Fallback: use last name first 3 letters or first 3 chars
                parts = s.split()
                if len(parts) > 1:
                    return parts[-1][:3].upper()
                return s[:3].upper()

            # Hozzunk létre egy új oszlopot rövidített driver-azonosítóval
            df_stints['DriverShort'] = df_stints['Driver'].apply(to_abbr)

            # Sorrendezés pozíció alapján az abbreviációk szerint
            sorted_drivers = df_stints.sort_values('Position')['DriverShort'].unique().tolist()
            
        else:
            # Fallback: lassú betöltés fastf1-gyel
            session = fastf1.get_session(year, round_number, 'R')
            session.load()
            laps = session.laps
            
            drivers = session.drivers
            stints_list = []
            
            # Versenyzők sorrendje a befutó alapján (ha elérhető)
            try:
                results = session.results
                sorted_drivers = results.sort_values(by='Position')['Abbreviation'].tolist()
            except:
                sorted_drivers = drivers

            for driver in drivers:
                driver_laps = laps.pick_drivers(driver)
                if driver_laps.empty:
                    continue
                
                # Stintek feldolgozása
                for stint_num, stint_laps in driver_laps.groupby('Stint'):
                    if stint_laps.empty:
                        continue
                        
                    compound = stint_laps['Compound'].iloc[0]
                    if pd.isna(compound) or compound == '':
                        compound = 'UNKNOWN'
                    
                    start_lap = stint_laps['LapNumber'].min()
                    end_lap = stint_laps['LapNumber'].max()
                    duration = end_lap - start_lap + 1
                    
                    stints_list.append({
                        'Driver': driver,
                        'Stint': stint_num,
                        'Compound': compound,
                        'StartLap': start_lap,
                        'EndLap': end_lap,
                        'Duration': duration
                    })
                    
            df_stints = pd.DataFrame(stints_list)
            event_name = session.event['EventName']
        
        if df_stints.empty:
            return go.Figure().update_layout(title="Nincs elérhető adat ehhez a futamhoz.")

        # Normalizáljuk a keverékeket (nagybetűsítve)
        df_stints['Compound'] = df_stints['Compound'].astype(str).str.upper()

        # Készítsünk dinamikus színtérképet: előre definiált közismert keverékekhez fix színek,
        # az ismeretlen/egyéb keverékekhez Plotly színskálát rendelünk.
        base_colors = {
            'SOFT': '#FF3333',
            'MEDIUM': '#FFF200',
            'HARD': '#EBEBEB',
            'INTERMEDIATE': '#39B54A',
            'WET': '#00AEEF',
            'UNKNOWN': '#808080'
        }

        unique_comps = list(df_stints['Compound'].unique())
        color_map = {}
        # Prepare a color palette for unknown compounds
        palette = px.colors.qualitative.Plotly
        palette_idx = 0

        for comp in unique_comps:
            if comp in base_colors:
                color_map[comp] = base_colors[comp]
            else:
                # assign next palette color
                color_map[comp] = palette[palette_idx % len(palette)]
                palette_idx += 1

        # Diagram készítése gyors, előfeldolgozott adatokból (javított, egységes stílus)
        try:
            max_lap = int(df_stints['EndLap'].max())
        except Exception:
            max_lap = None

        fig = px.bar(
            df_stints,
            x='Duration',
            y='DriverShort',
            base='StartLap',
            color='Compound',
            orientation='h',
            color_discrete_map=color_map,
            category_orders={'DriverShort': sorted_drivers[::-1]},
            title=f"Gumistratégiák - {event_name} {year}",
            hover_data={'Driver': True, 'StartLap': True, 'EndLap': True, 'Stint': True}
        )

        # Egységes megjelenés: vékony szegély, nagyobb kontraszt, enyhe átlátszóság
        fig.update_traces(marker_line_width=0.6, marker_line_color='rgba(0,0,0,0.6)', opacity=0.96)

        # X tengely beállítások
        xaxis = dict(title='Körszám', showgrid=True, gridcolor='rgba(255,255,255,0.06)')
        if max_lap is not None:
            xaxis['range'] = [0.5, max_lap + 0.5]
            xaxis['dtick'] = 1

        # Dinamikus magasság a versenyzők számához igazítva, jobb margók a hosszú nevekhez
        height = max(450, 36 * len(sorted_drivers))

        fig.update_layout(
            xaxis=xaxis,
            yaxis=dict(title='Versenyző (3 betűs)', categoryorder='array', categoryarray=sorted_drivers[::-1], automargin=True),
            legend=dict(title='Gumi Keverék', traceorder='normal'),
            bargap=0.16,
            bargroupgap=0.0,
            template='plotly_dark',
            height=height,
            margin=dict(l=120, r=40, t=80, b=80)
        )

        return fig

    except Exception as e:
        print(f"Hiba történt: {e}")
        return go.Figure().update_layout(title=f"Hiba az adatok betöltésekor: {str(e)}")

@app.callback(
    Output('qual-team-dropdown', 'disabled'),
    Input('qual-driver-dropdown', 'value')
)
def disable_team_dropdown(selected_drivers):
    if selected_drivers:
        return True
    return False

@app.callback(
    Output('qual-graph', 'figure'),
    [Input('qual-driver-dropdown', 'value'),
     Input('qual-race-dropdown', 'value'),
     Input('qual-team-dropdown', 'value'),
     Input('year-selector', 'value')]
)
def update_qual_graph(selected_drivers, selected_races, selected_teams, years):
    if not selected_races:
        return go.Figure().update_layout(title="Válassz legalább egy futamot!", template='plotly_dark')

    all_filtered_df = []
    
    # Group selected races by year
    races_by_year = {}
    for r_val in selected_races:
        try:
            y, r = map(int, str(r_val).split('_'))
            if y not in races_by_year:
                races_by_year[y] = []
            races_by_year[y].append(r)
        except ValueError:
            continue
            
    for year, rounds in races_by_year.items():
        _, df_results = get_data(year)
        if df_results.empty:
            continue
        
        filtered = df_results[df_results['RoundNumber'].isin(rounds)].copy()
        # Add Year to EventName to distinguish same race in different years
        filtered['EventName'] = filtered['EventName'] + f" ({year})"
        filtered['Year'] = year
        all_filtered_df.append(filtered)
        
    if not all_filtered_df:
         return go.Figure().update_layout(title="Nincs elérhető adat.", template='plotly_dark')
         
    filtered_df = pd.concat(all_filtered_df)
    
    # Szűrés versenyzőkre vagy csapatokra
    if selected_drivers:
        filtered_df = filtered_df[filtered_df['Abbreviation'].isin(selected_drivers)]
    elif selected_teams:
        team_col = 'TeamName' if 'TeamName' in filtered_df.columns else ('Team' if 'Team' in filtered_df.columns else None)
        if team_col:
            filtered_df = filtered_df[filtered_df[team_col].isin(selected_teams)]
        else:
             return go.Figure().update_layout(title="Hiányzó csapat információ az adatokban.", template='plotly_dark')
    
    if filtered_df.empty:
        return go.Figure().update_layout(title="Nincs adat a kiválasztott szűréshez.", template='plotly_dark')
        
    # Rendezés RoundNumber szerint
    filtered_df = filtered_df.sort_values(['Year', 'RoundNumber'])
    
    # GridPosition használata kvalifikációs eredményként
    y_col = 'GridPosition'
    if y_col not in filtered_df.columns:
        y_col = 'Position' # Fallback
        
    # Biztosítsuk, hogy numerikus
    filtered_df[y_col] = pd.to_numeric(filtered_df[y_col], errors='coerce')
    
    # Hover adatok összeállítása
    hover_data = {}
    if 'Q1' in filtered_df.columns: hover_data['Q1'] = True
    if 'Q2' in filtered_df.columns: hover_data['Q2'] = True
    if 'Q3' in filtered_df.columns: hover_data['Q3'] = True
    
    fig = px.line(
        filtered_df,
        x='EventName',
        y=y_col,
        color='Abbreviation',
        markers=True,
        title="Rajthelyek alakulása a szezon során",
        hover_data=hover_data
    )
    
    # Fordított Y tengely (1. hely felül)
    fig.update_yaxes(autorange="reversed", title="Rajthely")
    fig.update_xaxes(title="Futam")
    fig.update_layout(template='plotly_dark', autosize=True, margin=dict(l=40, r=40, t=40, b=40))
    
    return fig

@app.callback(
    Output('race-team-dropdown', 'disabled'),
    Input('race-driver-dropdown', 'value')
)
def disable_race_team_dropdown(selected_drivers):
    if selected_drivers:
        return True
    return False

@app.callback(
    Output('race-graph', 'figure'),
    [Input('race-driver-dropdown', 'value'),
     Input('race-race-dropdown', 'value'),
     Input('race-team-dropdown', 'value'),
     Input('year-selector', 'value')]
)
def update_race_graph(selected_drivers, selected_races, selected_teams, years):
    if not selected_races:
        return go.Figure().update_layout(title="Válassz legalább egy futamot!", template='plotly_dark')

    all_filtered_df = []
    
    # Group selected races by year
    races_by_year = {}
    for r_val in selected_races:
        try:
            y, r = map(int, str(r_val).split('_'))
            if y not in races_by_year:
                races_by_year[y] = []
            races_by_year[y].append(r)
        except ValueError:
            continue
            
    for year, rounds in races_by_year.items():
        _, df_results = get_data(year)
        if df_results.empty:
            continue
        
        filtered = df_results[df_results['RoundNumber'].isin(rounds)].copy()
        filtered['EventName'] = filtered['EventName'] + f" ({year})"
        filtered['Year'] = year
        all_filtered_df.append(filtered)
        
    if not all_filtered_df:
         return go.Figure().update_layout(title="Nincs elérhető adat.", template='plotly_dark')
         
    filtered_df = pd.concat(all_filtered_df)
    
    # Szűrés versenyzőkre vagy csapatokra
    if selected_drivers:
        filtered_df = filtered_df[filtered_df['Abbreviation'].isin(selected_drivers)]
    elif selected_teams:
        team_col = 'TeamName' if 'TeamName' in filtered_df.columns else ('Team' if 'Team' in filtered_df.columns else None)
        if team_col:
            filtered_df = filtered_df[filtered_df[team_col].isin(selected_teams)]
        else:
             return go.Figure().update_layout(title="Hiányzó csapat információ az adatokban.", template='plotly_dark')
    
    if filtered_df.empty:
        return go.Figure().update_layout(title="Nincs adat a kiválasztott szűréshez.", template='plotly_dark')
        
    # Rendezés RoundNumber szerint
    filtered_df = filtered_df.sort_values(['Year', 'RoundNumber'])
    
    # Position használata verseny eredményként
    y_col = 'Position'
        
    # Biztosítsuk, hogy numerikus
    filtered_df[y_col] = pd.to_numeric(filtered_df[y_col], errors='coerce')
    
    # Hover adatok összeállítása
    hover_data = {}
    if 'GridPosition' in filtered_df.columns: hover_data['GridPosition'] = True
    if 'Points' in filtered_df.columns: hover_data['Points'] = True
    if 'Status' in filtered_df.columns: hover_data['Status'] = True
    
    fig = px.line(
        filtered_df,
        x='EventName',
        y=y_col,
        color='Abbreviation',
        markers=True,
        title="Versenyeredmények alakulása a szezon során",
        hover_data=hover_data
    )
    
    # Fordított Y tengely (1. hely felül)
    fig.update_yaxes(autorange="reversed", title="Helyezés")
    fig.update_xaxes(title="Futam")
    fig.update_layout(template='plotly_dark', autosize=True, margin=dict(l=40, r=40, t=40, b=40))
    
    return fig

@app.callback(
    Output('champ-team-dropdown', 'disabled'),
    Input('champ-driver-dropdown', 'value')
)
def disable_champ_team_dropdown(selected_drivers):
    if selected_drivers:
        return True
    return False

@app.callback(
    Output('champ-graph', 'figure'),
    [Input('champ-driver-dropdown', 'value'),
     Input('champ-team-dropdown', 'value'),
     Input('year-selector', 'value')]
)
def update_champ_graph(selected_drivers, selected_teams, years):
    if not isinstance(years, list):
        years = [years]
        
    all_df = []
    for year in sorted(years):
        _, df_results = get_data(year)
        if df_results.empty:
            continue
        
        df = df_results.copy()
        if 'Points' not in df.columns:
            continue
            
        df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
        
        # Szűrés
        if selected_drivers:
            df = df[df['Abbreviation'].isin(selected_drivers)]
        elif selected_teams:
            team_col = 'TeamName' if 'TeamName' in df.columns else ('Team' if 'Team' in df.columns else None)
            if team_col:
                df = df[df[team_col].isin(selected_teams)]
        
        if df.empty:
            continue
            
        # Kumulatív pontszámítás per year
        df = df.sort_values('RoundNumber')
        df['CumulativePoints'] = df.groupby('Abbreviation')['Points'].cumsum()
        df['EventName'] = df['EventName'] + f" ({year})"
        df['Year'] = year
        
        # If multiple years, append year to driver name
        if len(years) > 1:
            df['Abbreviation'] = df['Abbreviation'] + f" '{str(year)[2:]}"
            
        all_df.append(df)
        
    if not all_df:
        return go.Figure().update_layout(title="Nincs elérhető adat.", template='plotly_dark')
        
    df_final = pd.concat(all_df)
    df_final = df_final.sort_values(['Year', 'RoundNumber'])
    
    hover_data = {'Points': True, 'Position': True}
    
    fig = px.line(
        df_final,
        x='EventName',
        y='CumulativePoints',
        color='Abbreviation',
        markers=True,
        title="Világbajnoki Pontverseny Alakulása",
        hover_data=hover_data
    )
    
    fig.update_yaxes(title="Összpontszám")
    fig.update_xaxes(title="Futam")
    fig.update_layout(template='plotly_dark', autosize=True, margin=dict(l=40, r=40, t=40, b=40))
    
    return fig

@app.callback(
    Output('gain-team-dropdown', 'disabled'),
    Input('gain-driver-dropdown', 'value')
)
def disable_gain_team_dropdown(selected_drivers):
    if selected_drivers:
        return True
    return False

@app.callback(
    Output('gain-graph', 'figure'),
    [Input('gain-driver-dropdown', 'value'),
     Input('gain-race-dropdown', 'value'),
     Input('gain-team-dropdown', 'value'),
     Input('year-selector', 'value')]
)
def update_gain_loss_graph(selected_drivers, selected_races, selected_teams, years):
    if not selected_races:
        return go.Figure().update_layout(title="Válassz legalább egy futamot!", template='plotly_dark')

    all_filtered_df = []
    
    # Group selected races by year
    races_by_year = {}
    for r_val in selected_races:
        try:
            y, r = map(int, str(r_val).split('_'))
            if y not in races_by_year:
                races_by_year[y] = []
            races_by_year[y].append(r)
        except ValueError:
            continue
            
    for year, rounds in races_by_year.items():
        _, df_results = get_data(year)
        if df_results.empty:
            continue
        
        filtered = df_results[df_results['RoundNumber'].isin(rounds)].copy()
        filtered['EventName'] = filtered['EventName'] + f" ({year})"
        filtered['Year'] = year
        all_filtered_df.append(filtered)
        
    if not all_filtered_df:
         return go.Figure().update_layout(title="Nincs elérhető adat.", template='plotly_dark')
         
    filtered_df = pd.concat(all_filtered_df)
    
    # Szűrés versenyzőkre vagy csapatokra
    if selected_drivers:
        filtered_df = filtered_df[filtered_df['Abbreviation'].isin(selected_drivers)]
    elif selected_teams:
        team_col = 'TeamName' if 'TeamName' in filtered_df.columns else ('Team' if 'Team' in filtered_df.columns else None)
        if team_col:
            filtered_df = filtered_df[filtered_df[team_col].isin(selected_teams)]
        else:
             return go.Figure().update_layout(title="Hiányzó csapat információ az adatokban.", template='plotly_dark')
    
    if filtered_df.empty:
        return go.Figure().update_layout(title="Nincs adat a kiválasztott szűréshez.", template='plotly_dark')
        
    # Rendezés RoundNumber szerint
    filtered_df = filtered_df.sort_values(['Year', 'RoundNumber'])
    
    # Adatok előkészítése
    # GridPosition és Position numerikussá alakítása
    filtered_df['GridPosition'] = pd.to_numeric(filtered_df['GridPosition'], errors='coerce').fillna(0)
    filtered_df['Position'] = pd.to_numeric(filtered_df['Position'], errors='coerce').fillna(0)
    
    # Pozíció változás számítása: (Rajthely - Befutó)
    # Pozitív érték = javított (pl. 10. helyről 5.-re: 10 - 5 = +5)
    # Negatív érték = rontott (pl. 2. helyről 4.-re: 2 - 4 = -2)
    filtered_df['PositionChange'] = filtered_df['GridPosition'] - filtered_df['Position']
    
    # Színkódolás: Zöld ha javított (>0), Piros ha rontott (<0), Szürke ha 0
    filtered_df['Color'] = filtered_df['PositionChange'].apply(
        lambda x: '#00ff00' if x > 0 else ('#ff0000' if x < 0 else '#808080')
    )
    
    # Hover adatok
    hover_data = {'GridPosition': True, 'Position': True, 'PositionChange': True}
    
    # Oszlopdiagram (Bar Chart)
    # Mivel több futam és több versenyző lehet, a csoportosított oszlopdiagram a legjobb
    fig = px.bar(
        filtered_df,
        x='EventName',
        y='PositionChange',
        color='Abbreviation', # Csoportosítás versenyzőnként
        barmode='group',
        title="Pozíció Nyereség/Veszteség (Rajthely vs. Befutó)",
        hover_data=hover_data
    )
    
    fig.update_yaxes(title="Pozíció Változás (+ Javított, - Rontott)")
    fig.update_xaxes(title="Futam")
    fig.update_layout(template='plotly_dark', autosize=True, margin=dict(l=40, r=40, t=40, b=40))
    
    return fig

@app.callback(
    Output('lap-team-dropdown', 'disabled'),
    Input('lap-driver-dropdown', 'value')
)
def disable_lap_team_dropdown(selected_drivers):
    if selected_drivers:
        return True
    return False

@app.callback(
    Output('lap-graph', 'figure'),
    [Input('lap-race-dropdown', 'value'),
     Input('lap-driver-dropdown', 'value'),
     Input('lap-team-dropdown', 'value'),
     Input('year-selector', 'value')]
)
def update_lap_dist_graph(race_value, selected_drivers, selected_teams, years):
    if not race_value:
        return go.Figure().update_layout(title="Válassz egy futamot!", template='plotly_dark')
    
    try:
        year, round_number = map(int, str(race_value).split('_'))
    except ValueError:
        return go.Figure().update_layout(title="Érvénytelen futam kiválasztás.", template='plotly_dark')

    try:
        # Betöltjük a session-t a kiválasztott futamhoz
        session = fastf1.get_session(year, round_number, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        
        # Csak a gyors köröket vesszük figyelembe (boxkiállás nélküli körök)
        laps = laps.pick_wo_box()
        
        # Szűrés versenyzőkre vagy csapatokra
        if selected_drivers:
            # Ha van kiválasztott versenyző, szűrünk rájuk
            # A fastf1-ben a 'Driver' oszlop tartalmazza az abbreviációt
            laps = laps[laps['Driver'].isin(selected_drivers)]
        elif selected_teams:
            # Ha nincs versenyző, de van csapat, szűrünk a csapatokra
            laps = laps[laps['Team'].isin(selected_teams)]
        
        if laps.empty:
            return go.Figure().update_layout(title="Nincs elérhető köridő adat a kiválasztott szűréshez.", template='plotly_dark')
        
        # Köridők konvertálása másodpercre
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        
        # Kiugró értékek szűrése (pl. Safety Car, sárga zászló miatti lassú körök)
        # Csak azokat a köröket tartjuk meg, amelyek a leggyorsabb kör 107%-án belül vannak (vagy egy ésszerű határ)
        # Vagy egyszerűen a boxplot majd kezeli a kiugrókat, de a nagyon lassú körök torzíthatják a skálát.
        # Egy egyszerű szűrés: < 1.2 * átlagos köridő (vagy medián)
        median_lap = laps['LapTimeSeconds'].median()
        laps_filtered = laps[laps['LapTimeSeconds'] < median_lap * 1.15] # 15%-kal lassabb körökig
        
        if laps_filtered.empty:
             laps_filtered = laps # Ha túl szigorú volt a szűrés, használjuk az eredetit
        
        # Box Plot készítése
        fig = px.box(
            laps_filtered,
            x='Driver',
            y='LapTimeSeconds',
            color='Team', # Színezés csapat szerint
            title=f"Köridők Eloszlása - {session.event['EventName']} {year}",
            points="all", # Minden pont megjelenítése (opcionális, lehet 'outliers' is)
            hover_data=['LapNumber', 'Compound']
        )
        
        fig.update_yaxes(title="Köridő (másodperc)")
        fig.update_xaxes(title="Versenyző")
        fig.update_layout(template='plotly_dark', autosize=True, margin=dict(l=40, r=40, t=40, b=40))
        
        return fig

    except Exception as e:
        print(f"Hiba a köridők betöltésekor: {e}")
        return go.Figure().update_layout(title=f"Hiba az adatok betöltésekor: {str(e)}", template='plotly_dark')

@app.callback(
    [Output('race-dropdown', 'options'),
     Output('race-dropdown', 'value'),
     Output('qual-race-dropdown', 'options'),
     Output('race-race-dropdown', 'options'),
     Output('gain-race-dropdown', 'options'),
     Output('lap-race-dropdown', 'options'),
     Output('lap-race-dropdown', 'value'),
     Output('qual-team-dropdown', 'options'),
     Output('qual-driver-dropdown', 'options'),
     Output('race-team-dropdown', 'options'),
     Output('race-driver-dropdown', 'options'),
     Output('champ-team-dropdown', 'options'),
     Output('champ-driver-dropdown', 'options'),
     Output('gain-team-dropdown', 'options'),
     Output('gain-driver-dropdown', 'options'),
     Output('lap-team-dropdown', 'options'),
     Output('lap-driver-dropdown', 'options')],
    Input('year-selector', 'value')
)
def update_all_options(years):
    race_options, driver_options, team_options = get_options(years)
    
    default_race_val = race_options[0]['value'] if race_options else None
    
    return (
        race_options, default_race_val,
        race_options, race_options, race_options, race_options, default_race_val,
        team_options, driver_options,
        team_options, driver_options,
        team_options, driver_options,
        team_options, driver_options,
        team_options, driver_options
    )

if __name__ == '__main__':
    app.run(debug=True)
