import panel as pn
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import fastf1
import folium
import math
from functools import lru_cache


# Initialize Panel extension with Plotly and specific design template
pn.extension('plotly', 'folium', design='material', theme='dark')

# --- Globális Beállítások és Adatkezelő Függvények ---

# Cache beállítása - engedélyezve
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

# Globális cache az adatoknak
DATA_CACHE = {}
SCHEDULE_CACHE = {}

# Default year for selectors
DEFAULT_YEAR = 2022

# Circuit Coordinates (Approximate)
CIRCUIT_LOCATIONS = {
    'Sakhir': (26.0325, 50.5106),
    'Jeddah': (21.6319, 39.1044),
    'Melbourne': (-37.8497, 144.968),
    'Baku': (40.3725, 49.8533),
    'Miami': (25.958, -80.2389),
    'Imola': (44.3439, 11.7167),
    'Monaco': (43.7347, 7.4206),
    'Barcelona': (41.57, 2.2611),
    'Montréal': (45.5017, -73.5673),
    'Montreal': (45.5017, -73.5673),
    'Spielberg': (47.2197, 14.7647),
    'Silverstone': (52.0786, -1.0169),
    'Budapest': (47.583, 19.250),
    'Spa-Francorchamps': (50.4372, 5.9714),
    'Zandvoort': (52.3888, 4.5409),
    'Monza': (45.6156, 9.2811),
    'Singapore': (1.2914, 103.864),
    'Suzuka': (34.8431, 136.541),
    'Lusail': (25.4888, 51.4542),
    'Austin': (30.1328, -97.6411),
    'Mexico City': (19.4042, -99.0907),
    'São Paulo': (-23.7036, -46.6997),
    'Sao Paulo': (-23.7036, -46.6997),
    'Las Vegas': (36.1147, -115.173),
    'Yas Island': (24.4672, 54.6031),
    'Abu Dhabi': (24.4672, 54.6031),
    'Shanghai': (31.3389, 121.22),
    'Portimão': (37.227, -8.628),
    'Istanbul': (40.9517, 29.405),
    'Le Castellet': (43.2506, 5.7917),
    'Sochi': (43.4056, 39.9578),
    'Nürburg': (50.3356, 6.9475),
    'Mugello': (43.9975, 11.3719),
    'Doha': (25.4888, 51.4542)
}

def get_data(year):
    if year is None:
        year = DEFAULT_YEAR
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

# A fastf1 hívást gyorsítótárazzuk, mivel lassú lehet
@lru_cache(maxsize=16)
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
    
    all_race_options = {} # Changed to dict for Panel {label: value}
    drivers_set = set()
    teams_set = set()
    driver_labels = {}
    
    for year in sorted(years):
        # Races
        schedule = get_schedule_data(year)
        if not schedule.empty:
            for _, row in schedule.iterrows():
                label = f"{year} Round {row['RoundNumber']}: {row['EventName']}"
                value = f"{year}_{row['RoundNumber']}"
                all_race_options[label] = value
        
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
            except Exception:
                pass

    # Sort and format
    team_options = sorted(list(teams_set))
    # Driver options as dict {label: value}
    driver_options = {driver_labels.get(d, d): d for d in sorted(list(drivers_set))}
            
    return all_race_options, driver_options, team_options

# --- Grafikonokat Generáló Függvények ---

def create_tire_graph(race_value, year_selector_value):
    """Létrehozza a gumistratégia Plotly ábrát."""
    # Handle list input from CheckBoxGroup (take first selected)
    if isinstance(race_value, list):
        race_value = race_value[0] if race_value else None

    if not race_value:
        return go.Figure().update_layout(title="Válassz egy futamot!", template='plotly_dark')
    
    try:
        year, round_number = map(int, str(race_value).split('_'))
    except ValueError:
        return go.Figure().update_layout(title="Érvénytelen futam kiválasztás.", template='plotly_dark')

    df_all_data, df_results = get_data(year)

    
    if df_all_data.empty:
         return go.Figure().update_layout(title="Nincs elérhető adat ehhez az évhez.", template='plotly_dark')

    df_stints = df_all_data[df_all_data['RoundNumber'] == round_number].copy()
            
    if df_stints.empty:
        return go.Figure().update_layout(title="Nincs elérhető adat ehhez a futamhoz a CSV-ben.", template='plotly_dark')
            
    event_name = df_stints['EventName'].iloc[0]
            
    # Map Driver to Abbreviation if possible
    driver_map = {}
    if not df_results.empty:
        df_r = df_results[df_results['RoundNumber'] == round_number].copy()
        
        # Helper to normalize numbers (handle 44.0 vs 44 vs "44")
        def normalize_key(x):
            try:
                return str(int(float(x)))
            except (ValueError, TypeError):
                return str(x).strip()

        if 'Abbreviation' in df_r.columns:
            # Map from Number (Standard FastF1 name)
            if 'Number' in df_r.columns:
                keys = df_r['Number'].apply(normalize_key)
                driver_map.update(dict(zip(keys, df_r['Abbreviation'])))
            
            # Map from DriverNumber (Alternative name seen in Excel)
            if 'DriverNumber' in df_r.columns:
                keys = df_r['DriverNumber'].apply(normalize_key)
                driver_map.update(dict(zip(keys, df_r['Abbreviation'])))

            # Map from Driver Name (just in case)
            if 'Driver' in df_r.columns:
                driver_map.update(dict(zip(df_r['Driver'].astype(str), df_r['Abbreviation'])))


    # Apply mapping
    # Normalize the stint driver identifiers too
    def normalize_stint_driver(x):
        try:
            return str(int(float(x)))
        except (ValueError, TypeError):
            return str(x).strip()
            
    stint_drivers_normalized = df_stints['Driver'].apply(normalize_stint_driver)
    df_stints['DriverShort'] = stint_drivers_normalized.map(driver_map)
    
    # Fallback for unmapped drivers
    mask = df_stints['DriverShort'].isna()
    if mask.any():
        # Try to use the original value if mapping failed
        df_stints.loc[mask, 'DriverShort'] = (
            df_stints.loc[mask, 'Driver'].astype(str)
            .str.split()
            .str[-1]
            .str[:3]
            .str.upper()
        )
    
    # Sorrendezés pozíció alapján
    sorted_drivers = []
    if not df_results.empty:
        df_r = df_results[df_results['RoundNumber'] == round_number]
        for driver in df_r.sort_values('Position')['Abbreviation'].unique():
             if driver in df_stints['DriverShort'].unique():
                 sorted_drivers.append(driver)
        
    if not sorted_drivers:
         sorted_drivers = df_stints['DriverShort'].unique().tolist()
    
    df_stints['Compound'] = df_stints['Compound'].astype(str).str.upper()

    # Színtérkép
    base_colors = {
        'SOFT': '#FF3333', 'MEDIUM': '#FFF200', 'HARD': '#EBEBEB',
        'INTERMEDIATE': '#39B54A', 'WET': '#00AEEF', 'UNKNOWN': '#808080'
    }
    unique_comps = list(df_stints['Compound'].unique())
    color_map = {}
    palette = px.colors.qualitative.Plotly
    palette_idx = 0

    for comp in unique_comps:
        if comp in base_colors:
            color_map[comp] = base_colors[comp]
        else:
            color_map[comp] = palette[palette_idx % len(palette)]
            palette_idx += 1

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

    fig.update_traces(marker_line_width=0.6, marker_line_color='rgba(0,0,0,0.6)', opacity=0.96)
    xaxis = dict(title='Körszám', showgrid=True, gridcolor='rgba(255,255,255,0.06)')
    if max_lap is not None:
        xaxis['range'] = [0.5, max_lap + 0.5]

    fig.update_layout(
        xaxis=xaxis,
        yaxis=dict(title='Versenyző', categoryorder='array', categoryarray=sorted_drivers[::-1], automargin=True),
        legend=dict(title='Gumi Keverék', traceorder='normal'),
        bargap=0.16,
        template='plotly_dark',
        height=450,
        margin=dict(l=120, r=40, t=80, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def create_qual_graph(selected_drivers, selected_races, selected_teams, years):
    """Létrehozza a kvalifikációs eredmények Plotly ábráját."""
    if not selected_races:
        return go.Figure().update_layout(title="Válassz legalább egy futamot!", template='plotly_dark')

    all_filtered_df = []
    races_by_year = {}
    for r_val in selected_races:
        try:
            y, r = map(int, str(r_val).split('_'))
            if y not in races_by_year: races_by_year[y] = []
            races_by_year[y].append(r)
        except ValueError:
            continue

    for year, rounds in races_by_year.items():
        _, df_results = get_data(year)
        if df_results.empty:
            continue

        filtered = df_results[df_results['RoundNumber'].isin(rounds)].copy()
        filtered['Year'] = year
        all_filtered_df.append(filtered)

    if not all_filtered_df:
        return go.Figure().update_layout(title="Nincs elérhető adat.", template='plotly_dark')

    filtered_df = pd.concat(all_filtered_df)

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

    filtered_df = filtered_df.sort_values(['Year', 'RoundNumber'])
    y_col = 'GridPosition'
    if y_col not in filtered_df.columns:
        y_col = 'Position'
    filtered_df[y_col] = pd.to_numeric(filtered_df[y_col], errors='coerce')

    hover_data = {}
    if 'Q1' in filtered_df.columns:
        hover_data['Q1'] = True
    if 'Q2' in filtered_df.columns:
        hover_data['Q2'] = True
    if 'Q3' in filtered_df.columns:
        hover_data['Q3'] = True

    round_label_map = filtered_df.groupby('RoundNumber')['EventName'].first().to_dict()
    filtered_df['Year'] = filtered_df['Year'].astype(str)

    if selected_drivers and len(selected_drivers) == 1:
        fig = px.line(
            filtered_df, x='RoundNumber', y=y_col, color='Year', markers=True,
            title="Rajthelyek alakulása a szezon során", hover_data=hover_data
        )
    else:
        fig = px.line(
            filtered_df, x='RoundNumber', y=y_col, color='Abbreviation', line_dash='Year', markers=True,
            title="Rajthelyek alakulása a szezon során", hover_data=hover_data
        )

    tick_vals = sorted(round_label_map.keys())
    tick_texts = [round_label_map[r] for r in tick_vals]
    fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_texts)

    fig.update_yaxes(autorange="reversed", title="Rajthely")
    fig.update_xaxes(title="Futam")
    fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=120, r=40, t=80, b=80), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig

def create_race_graph(selected_drivers, selected_races, selected_teams, years):
    """Létrehozza a verseny eredmények Plotly ábráját."""
    if not selected_races:
        return go.Figure().update_layout(title="Válassz legalább egy futamot!", template='plotly_dark')

    all_filtered_df = []
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
        filtered['Year'] = year
        all_filtered_df.append(filtered)

    if not all_filtered_df:
        return go.Figure().update_layout(title="Nincs elérhető adat.", template='plotly_dark')

    filtered_df = pd.concat(all_filtered_df)

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

    filtered_df = filtered_df.sort_values(['Year', 'RoundNumber'])
    y_col = 'Position'
    filtered_df[y_col] = pd.to_numeric(filtered_df[y_col], errors='coerce')

    hover_data = {}
    if 'GridPosition' in filtered_df.columns:
        hover_data['GridPosition'] = True
    if 'Points' in filtered_df.columns:
        hover_data['Points'] = True
    if 'Status' in filtered_df.columns:
        hover_data['Status'] = True

    round_label_map = filtered_df.groupby('RoundNumber')['EventName'].first().to_dict()
    filtered_df['Year'] = filtered_df['Year'].astype(str)

    if selected_drivers and len(selected_drivers) == 1:
        fig = px.line(
            filtered_df, x='RoundNumber', y=y_col, color='Year', markers=True,
            title="Versenyeredmények alakulása a szezon során", hover_data=hover_data
        )
    else:
        fig = px.line(
            filtered_df, x='RoundNumber', y=y_col, color='Abbreviation', line_dash='Year', markers=True,
            title="Versenyeredmények alakulása a szezon során", hover_data=hover_data
        )

    tick_vals = sorted(round_label_map.keys())
    tick_texts = [round_label_map[r] for r in tick_vals]
    fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_texts)

    fig.update_yaxes(autorange="reversed", title="Helyezés")
    fig.update_xaxes(title="Futam")
    fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=120, r=40, t=80, b=80), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig

def create_champ_graph(selected_drivers, selected_teams, years):
    """Létrehozza a bajnoki pontverseny Plotly ábráját."""
    if not isinstance(years, list): years = [years]
        
    all_df = []
    for year in sorted(years):
        _, df_results = get_data(year)
        if df_results.empty: continue
        
        df = df_results.copy()
        if 'Points' not in df.columns: continue
            
        df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
        
        if selected_drivers:
            df = df[df['Abbreviation'].isin(selected_drivers)]
        elif selected_teams:
            team_col = 'TeamName' if 'TeamName' in df.columns else ('Team' if 'Team' in df.columns else None)
            if team_col:
                df = df[df[team_col].isin(selected_teams)]
        
        if df.empty: continue
            
        df = df.sort_values('RoundNumber')
        df['CumulativePoints'] = df.groupby('Abbreviation')['Points'].cumsum()
        df['Year'] = year
            
        all_df.append(df)
        
    if not all_df: return go.Figure().update_layout(title="Nincs elérhető adat.", template='plotly_dark')
        
    df_final = pd.concat(all_df)
    df_final = df_final.sort_values(['Year', 'RoundNumber'])
    df_final['Year'] = df_final['Year'].astype(str)

    hover_data = {'Points': True, 'Position': True, 'Abbreviation': True, 'Year': True}
    round_label_map = df_final.groupby('RoundNumber')['EventName'].first().to_dict()

    if selected_drivers and len(selected_drivers) == 1:
        fig = px.line(
            df_final, x='RoundNumber', y='CumulativePoints', color='Year', markers=True,
            title="Világbajnoki Pontverseny Alakulása", hover_data=hover_data
        )
    else:
        fig = px.line(
            df_final, x='RoundNumber', y='CumulativePoints', color='Abbreviation', line_dash='Year', markers=True,
            title="Világbajnoki Pontverseny Alakulása", hover_data=hover_data
        )

    tick_vals = sorted(round_label_map.keys())
    tick_texts = [round_label_map[r] for r in tick_vals]
    fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_texts)
    
    fig.update_yaxes(title="Összpontszám")
    fig.update_xaxes(title="Futam")
    fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=120, r=40, t=80, b=80), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    return fig

def create_gain_loss_graph(selected_drivers, selected_races, selected_teams, years):
    """Létrehozza a pozíció változás Plotly ábráját."""
    if not selected_races:
        return go.Figure().update_layout(title="Válassz legalább egy futamot!", template='plotly_dark')

    all_filtered_df = []
    races_by_year = {}
    for r_val in selected_races:
        try:
            y, r = map(int, str(r_val).split('_'))
            if y not in races_by_year: races_by_year[y] = []
            races_by_year[y].append(r)
        except ValueError: continue
            
    for year, rounds in races_by_year.items():
        _, df_results = get_data(year)
        if df_results.empty: continue
        
        filtered = df_results[df_results['RoundNumber'].isin(rounds)].copy()
        filtered['EventName'] = filtered['EventName'] + f" ({year})"
        filtered['Year'] = year
        all_filtered_df.append(filtered)
        
    if not all_filtered_df: return go.Figure().update_layout(title="Nincs elérhető adat.", template='plotly_dark')
         
    filtered_df = pd.concat(all_filtered_df)
    
    if selected_drivers:
        filtered_df = filtered_df[filtered_df['Abbreviation'].isin(selected_drivers)]
    elif selected_teams:
        team_col = 'TeamName' if 'TeamName' in filtered_df.columns else ('Team' if 'Team' in filtered_df.columns else None)
        if team_col:
            filtered_df = filtered_df[filtered_df[team_col].isin(selected_teams)]
        else: return go.Figure().update_layout(title="Hiányzó csapat információ az adatokban.", template='plotly_dark')
    
    if filtered_df.empty: return go.Figure().update_layout(title="Nincs adat a kiválasztott szűréshez.", template='plotly_dark')
        
    filtered_df = filtered_df.sort_values(['Year', 'RoundNumber'])
    
    filtered_df['GridPosition'] = pd.to_numeric(filtered_df['GridPosition'], errors='coerce').fillna(0)
    filtered_df['Position'] = pd.to_numeric(filtered_df['Position'], errors='coerce').fillna(0)
    filtered_df['PositionChange'] = filtered_df['GridPosition'] - filtered_df['Position']
    
    hover_data = {'GridPosition': True, 'Position': True, 'PositionChange': True}
    
    fig = px.bar(
        filtered_df, x='PositionChange', y='EventName', color='Abbreviation', orientation='h', barmode='group',
        title="Pozíció Nyereség/Veszteség (Rajthely vs. Befutó)", hover_data=hover_data
    )

    fig.update_xaxes(title="Pozíció Változás (+ Javított, - Rontott)")
    fig.update_yaxes(title="Futam", automargin=True)
    fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=220, r=40, t=80, b=80), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    return fig

def create_lap_dist_graph(race_value, selected_drivers, selected_teams, years):
    """Létrehozza a köridő eloszlás Plotly ábráját."""
    # Handle list input from CheckBoxGroup (take first selected)
    if isinstance(race_value, list):
        race_value = race_value[0] if race_value else None

    if not race_value:
        return go.Figure().update_layout(title="Válassz egy futamot!", template='plotly_dark')
    
    try:
        year, round_number = map(int, str(race_value).split('_'))
    except ValueError:
        return go.Figure().update_layout(title="Érvénytelen futam kiválasztás.", template='plotly_dark')

    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        
        laps = laps.pick_wo_box()
        
        if selected_drivers:
            laps = laps[laps['Driver'].isin(selected_drivers)]
        elif selected_teams:
            laps = laps[laps['Team'].isin(selected_teams)]
        
        if laps.empty:
            return go.Figure().update_layout(title="Nincs elérhető köridő adat a kiválasztott szűréshez.", template='plotly_dark')
        
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        
        median_lap = laps['LapTimeSeconds'].median()
        laps_filtered = laps[laps['LapTimeSeconds'] < median_lap * 1.15]
        if laps_filtered.empty: laps_filtered = laps
        
        fig = px.box(
            laps_filtered, x='Driver', y='LapTimeSeconds', color='Team',
            title=f"Köridők Eloszlása - {session.event['EventName']} {year}",
            points="all", hover_data=['LapNumber', 'Compound']
        )
        
        fig.update_yaxes(title="Köridő (másodperc)")
        fig.update_xaxes(title="Versenyző")
        fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=120, r=40, t=80, b=80), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        
        return fig

    except Exception as e:
        return go.Figure().update_layout(title=f"Hiba az adatok betöltésekor: {str(e)}", template='plotly_dark')

def create_map_graph(years, selected_race=None, target_hash_prefix='race'):
    """Létrehozza a világtérképet a futamhelyszínekkel (Folium)."""
    if not isinstance(years, list): years = [years]
    
    # Handle list input for selected_race
    if isinstance(selected_race, list):
        selected_race = selected_race[0] if selected_race else None
    
    all_schedules = []
    for year in years:
        sch = get_schedule_data(year)
        if not sch.empty:
            sch = sch.copy()
            sch['Year'] = str(year)
            all_schedules.append(sch)
            
    if not all_schedules:
        return folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
        
    df = pd.concat(all_schedules)
    
    # Map coordinates
    df['Lat'] = df['Location'].map(lambda x: CIRCUIT_LOCATIONS.get(x, (None, None))[0])
    df['Lon'] = df['Location'].map(lambda x: CIRCUIT_LOCATIONS.get(x, (None, None))[1])
    
    # Drop missing
    df = df.dropna(subset=['Lat', 'Lon'])
    
    # Create Folium map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")

    # Parse selected race
    selected_year = None
    selected_round = None
    if selected_race:
        try:
            selected_year, selected_round = map(int, str(selected_race).split('_'))
        except ValueError:
            pass

    for idx, row in df.iterrows():
        # Add a button to select the race
        select_js = f"window.top.location.hash = '{target_hash_prefix}={row['Year']}_{row['RoundNumber']}';"
        popup_text = f"""
        <div style="font-family: sans-serif; color: black;">
            <b>{row['EventName']}</b><br>
            Year: {row['Year']}<br>
            Round: {row['RoundNumber']}<br>
            Location: {row['Location']}, {row['Country']}<br>
            Date: {row['EventDate']}<br>
            <button onclick="{select_js}" style="margin-top: 5px; cursor: pointer; padding: 5px; background-color: #FF1801; color: white; border: none; border-radius: 3px;">Kiválasztás</button>
        </div>
        """
        
        # Check if this is the selected race
        is_selected = False
        if selected_year is not None and selected_round is not None:
            if int(row['Year']) == selected_year and int(row['RoundNumber']) == selected_round:
                is_selected = True
        
        chart_added = False
        if is_selected:
            # Generate Donut Chart SVG
            try:
                df_all_data, _ = get_data(selected_year)
                if not df_all_data.empty:
                    df_stints = df_all_data[df_all_data['RoundNumber'] == selected_round].copy()
                    if not df_stints.empty:
                        df_stints['Compound'] = df_stints['Compound'].astype(str).str.upper()
                        df_agg = df_stints.groupby('Compound')['Duration'].sum().to_dict()
                        
                        base_colors = {
                            'SOFT': '#FF3333', 'MEDIUM': '#FFF200', 'HARD': '#EBEBEB',
                            'INTERMEDIATE': '#39B54A', 'WET': '#00AEEF', 'UNKNOWN': '#808080'
                        }
                        
                        # SVG Generation
                        radius = 40
                        inner_radius = 20
                        total = sum(df_agg.values())
                        
                        svg_parts = []
                        size = radius * 2 + 4
                        svg_parts.append(f'<svg width="{size}" height="{size}" viewBox="-{radius+2} -{radius+2} {size} {size}" style="transform: translate(-50%, -50%);">')
                        
                        start_angle = 0
                        for compound, value in df_agg.items():
                            if value <= 0: continue
                            fraction = value / total
                            end_angle = start_angle + fraction * 2 * math.pi
                            
                            # Coordinates (0 is up)
                            x1_out = radius * math.sin(start_angle)
                            y1_out = -radius * math.cos(start_angle)
                            x2_out = radius * math.sin(end_angle)
                            y2_out = -radius * math.cos(end_angle)
                            
                            x1_in = inner_radius * math.sin(start_angle)
                            y1_in = -inner_radius * math.cos(start_angle)
                            x2_in = inner_radius * math.sin(end_angle)
                            y2_in = -inner_radius * math.cos(end_angle)
                            
                            large_arc = 1 if fraction > 0.5 else 0
                            
                            color = base_colors.get(compound, '#808080')
                            
                            if fraction > 0.999:
                                # Full circle
                                path = f'M 0 -{radius} A {radius} {radius} 0 1 1 0 {radius} A {radius} {radius} 0 1 1 0 -{radius} M 0 -{inner_radius} A {inner_radius} {inner_radius} 0 1 0 0 {inner_radius} A {inner_radius} {inner_radius} 0 1 0 0 -{inner_radius} Z'
                                svg_parts.append(f'<path d="{path}" fill="{color}" stroke="none" fill-rule="evenodd" />')
                            else:
                                path = f'M {x1_out} {y1_out} A {radius} {radius} 0 {large_arc} 1 {x2_out} {y2_out} L {x2_in} {y2_in} A {inner_radius} {inner_radius} 0 {large_arc} 0 {x1_in} {y1_in} Z'
                                svg_parts.append(f'<path d="{path}" fill="{color}" stroke="none" />')
                            
                            # Add Percentage Text
                            if fraction > 0.08: # Only show if slice is big enough
                                mid_angle = (start_angle + end_angle) / 2
                                text_r = (radius + inner_radius) / 2
                                tx = text_r * math.sin(mid_angle)
                                ty = -text_r * math.cos(mid_angle)
                                pct = int(round(fraction * 100))
                                svg_parts.append(f'<text x="{tx}" y="{ty}" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="10" font-family="sans-serif" font-weight="bold" style="text-shadow: 1px 1px 2px black;">{pct}%</text>')

                            start_angle = end_angle
                            
                        # Center Red Dot
                        svg_parts.append(f'<circle cx="0" cy="0" r="{inner_radius-5}" fill="#FF1801" stroke="white" stroke-width="2" />')
                        svg_parts.append('</svg>')
                        
                        svg_icon = "".join(svg_parts)
                        
                        folium.Marker(
                            location=[row['Lat'], row['Lon']],
                            icon=folium.DivIcon(html=svg_icon, icon_size=(size, size), icon_anchor=(size/2, size/2)),
                            popup=folium.Popup(popup_text, max_width=300),
                            tooltip=f"{row['EventName']} ({row['Year']})"
                        ).add_to(m)
                        chart_added = True
            except Exception as e:
                print(f"Error generating map chart: {e}")

        if not chart_added:
            folium.CircleMarker(
                location=[row['Lat'], row['Lon']],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{row['EventName']} ({row['Year']})",
                color="#FF1801", # F1 Red
                fill=True,
                fill_color="#FF1801",
                fill_opacity=0.7
            ).add_to(m)
    
    return m

def create_tire_distribution_chart(race_value, year_selector_value):
    """Létrehozza a gumihasználat kördiagramját (Donut Chart)."""
    # Handle list input
    if isinstance(race_value, list):
        race_value = race_value[0] if race_value else None

    if not race_value:
        return go.Figure().update_layout(title="Válassz egy futamot!", template='plotly_dark')
    
    try:
        year, round_number = map(int, str(race_value).split('_'))
    except ValueError:
        return go.Figure().update_layout(title="Érvénytelen futam kiválasztás.", template='plotly_dark')

    df_all_data, _ = get_data(year)
    
    if df_all_data.empty:
         return go.Figure().update_layout(title="Nincs elérhető adat ehhez az évhez.", template='plotly_dark')

    df_stints = df_all_data[df_all_data['RoundNumber'] == round_number].copy()
            
    if df_stints.empty:
        return go.Figure().update_layout(title="Nincs elérhető adat ehhez a futamhoz.", template='plotly_dark')
            
    event_name = df_stints['EventName'].iloc[0]
    
    # Aggregate data
    df_stints['Compound'] = df_stints['Compound'].astype(str).str.upper()
    
    # Filter out unknown or test compounds if necessary, but usually we keep them
    # Sum duration (laps) per compound
    df_agg = df_stints.groupby('Compound')['Duration'].sum().reset_index()
    df_agg.columns = ['Compound', 'TotalLaps']
    
    # Colors
    base_colors = {
        'SOFT': '#FF3333', 'MEDIUM': '#FFF200', 'HARD': '#EBEBEB',
        'INTERMEDIATE': '#39B54A', 'WET': '#00AEEF', 'UNKNOWN': '#808080'
    }
    
    # Map colors
    colors = [base_colors.get(c, '#808080') for c in df_agg['Compound']]
    
    fig = go.Figure(data=[go.Pie(
        labels=df_agg['Compound'],
        values=df_agg['TotalLaps'],
        hole=.4,
        marker=dict(colors=colors, line=dict(color='#000000', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title=f"Gumihasználat Eloszlása - {event_name} {year}",
        template='plotly_dark',
        height=450,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    return fig

def render_selected_plot(choice, selected_drivers, selected_races, selected_teams, years, side=None):
    """Generic renderer for the compare view."""
    # Handle list input from CheckBoxGroup (take first selected)
    if isinstance(choice, list):
        choice = choice[0] if choice else None

    if not choice:
        return go.Figure().update_layout(title="Válassz diagram típust", template='plotly_dark')

    if not isinstance(years, list):
        years = [years] if years is not None else [DEFAULT_YEAR]

    drivers = selected_drivers or []
    teams = selected_teams or []
    races = selected_races or []

    choice = str(choice)
    try:
        if choice == 'Gumistratégiák':
            race_val = races[0] if isinstance(races, (list, tuple)) and races else None
            return create_tire_graph(race_val, years)
        elif choice == 'Kvalifikáció':
            return create_qual_graph(drivers, races, teams, years)
        elif choice == 'Verseny':
            return create_race_graph(drivers, races, teams, years)
        elif choice == 'Bajnokság':
            return create_champ_graph(drivers, teams, years)
        elif choice == 'Pozíció Változás':
            return create_gain_loss_graph(drivers, races, teams, years)
        elif choice == 'Köridők':
            race_val = races[0] if isinstance(races, (list, tuple)) and races else None
            return create_lap_dist_graph(race_val, drivers, teams, years)
        elif choice == 'Térkép':
            race_val = races[0] if isinstance(races, (list, tuple)) and races else None
            prefix = 'race'
            if side == 'left': prefix = 'race_left'
            elif side == 'right': prefix = 'race_right'
            
            map_obj = create_map_graph(years, race_val, target_hash_prefix=prefix)
            
            return pn.pane.plot.Folium(map_obj, sizing_mode='stretch_width', height=450)
        else:
            return go.Figure().update_layout(title="Ismeretlen diagram típus.", template='plotly_dark')
    except Exception as e:
        return go.Figure().update_layout(title=f"Hiba a diagram létrehozásakor: {e}", template='plotly_dark')

# --- Panel UI Setup ---

# Custom CSS for React-like look
custom_css = """
:root {
    --f1-red: #FF1801;
    --card-bg: #1e1e1e;
    --page-bg: #121212;
    --text-color: #e0e0e0;
    --border-radius: 12px;
}

body {
    background-color: var(--page-bg);
    color: var(--text-color);
    font-family: 'Roboto', 'Segoe UI', sans-serif;
}

.bk-root .bk-tabs-header .bk-tab {
    background-color: transparent;
    color: #888;
    border: none;
    font-weight: 500;
    transition: color 0.3s ease;
}

.bk-root .bk-tabs-header .bk-tab.bk-active {
    color: var(--f1-red);
    border-bottom: 2px solid var(--f1-red);
}

.card-box {
    background-color: var(--card-bg);
    border-radius: var(--border-radius);
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    border: 1px solid #333;
    margin-bottom: 20px;
}

.widget-box {
    background-color: #252525;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    border-left: 4px solid var(--f1-red);
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #1a1a1a; 
}
::-webkit-scrollbar-thumb {
    background: #444; 
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #666; 
}

/* Dropdown Checkbox Styling */
.dropdown-btn button {
    text-align: left !important;
    justify-content: space-between !important;
    background-color: #333 !important;
    color: #e0e0e0 !important;
    border: 1px solid #555 !important;
}
.dropdown-btn button:hover {
    border-color: var(--f1-red) !important;
}
.dropdown-box {
    background-color: #2a2a2a;
    border: 1px solid #444;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 10px;
    margin-top: -5px;
    z-index: 100;
}
"""

pn.config.raw_css.append(custom_css)

# Initial Data
INIT_RACE_OPTIONS, INIT_DRIVER_OPTIONS, INIT_TEAM_OPTIONS = get_options([DEFAULT_YEAR])
INIT_RACE_VALUES = list(INIT_RACE_OPTIONS.values())

# Helper for Dropdown Checkbox
def create_dropdown_checkbox(name, options, value):
    # Ensure value is a list
    if value is None: value = []
    if not isinstance(value, list): value = [value]
    
    # 1. The "Source of Truth" widget (hidden)
    # We use MultiChoice because it holds list of values and options.
    # This is what the app logic interacts with (setting options, getting values).
    state_widget = pn.widgets.MultiChoice(name='state', options=options, value=value, visible=False)
    
    # 2. The UI widgets
    search_box = pn.widgets.TextInput(placeholder="Keresés...", sizing_mode='stretch_width', margin=(0, 0, 5, 0))
    
    # Helper to normalize options for filtering
    def get_opts_dict(opts):
        if isinstance(opts, list):
            return {str(v): v for v in opts}
        return opts

    opts_dict = get_opts_dict(options)
        
    # Initial CBG options (all)
    cbg = pn.widgets.CheckBoxGroup(name='', options=options, value=value, sizing_mode='stretch_width')
    
    # Toggle All Button
    toggle_all_btn = pn.widgets.Button(name="Összes kijelölése / Törlése", button_type='light', sizing_mode='stretch_width', margin=(0, 0, 5, 0), height=30)
    
    def on_toggle_all(event):
        # Get currently visible options
        current_opts = cbg.options
        visible_values = list(current_opts.values()) if isinstance(current_opts, dict) else current_opts
        
        if not visible_values: return

        # Check if all visible are currently selected
        current_selected_visible = [v for v in state_widget.value if v in visible_values]
        all_selected = len(current_selected_visible) == len(visible_values)
        
        if all_selected:
            # Deselect all visible
            new_state = [v for v in state_widget.value if v not in visible_values]
        else:
            # Select all visible
            hidden_selected = [v for v in state_widget.value if v not in visible_values]
            new_state = hidden_selected + visible_values
        
        state_widget.value = new_state

    toggle_all_btn.on_click(on_toggle_all)

    # Toggle Button
    btn = pn.widgets.Button(name=f"{name} ({len(value)}) ▼", button_type='default', css_classes=['dropdown-btn'])
    
    # Container
    container = pn.Column(search_box, toggle_all_btn, cbg, visible=False, max_height=300, scroll=True, css_classes=['dropdown-box'], sizing_mode='stretch_width')
    
    # Logic flags
    is_updating_cbg = False
    
    def toggle(event):
        container.visible = not container.visible
        icon = "▲" if container.visible else "▼"
        btn.name = f"{name} ({len(state_widget.value)}) {icon}"
        
    def on_search(event):
        nonlocal is_updating_cbg
        term = event.new.lower() if event.new else ""
        
        # Filter options
        if not term:
            new_opts = state_widget.options # Restore full options from state
        else:
            # Filter keys in opts_dict
            new_opts = {k: v for k, v in opts_dict.items() if term in str(k).lower()}
        
        is_updating_cbg = True
        cbg.options = new_opts
        
        # Update cbg value to match state_widget but only for visible options
        allowed_values = list(new_opts.values()) if isinstance(new_opts, dict) else new_opts
        cbg.value = [v for v in state_widget.value if v in allowed_values]
        is_updating_cbg = False
        
    def on_cbg_change(event):
        nonlocal is_updating_cbg
        if is_updating_cbg: return
        
        # User clicked a checkbox -> Update state_widget.value
        current_visible_opts = cbg.options
        visible_values = list(current_visible_opts.values()) if isinstance(current_visible_opts, dict) else current_visible_opts
        
        # Items in state that are NOT currently visible (should be preserved)
        hidden_selected = [v for v in state_widget.value if v not in visible_values]
        
        # New state
        new_state = hidden_selected + event.new
        
        # Update state widget
        state_widget.value = new_state
        
        # Update button label
        icon = "▲" if container.visible else "▼"
        btn.name = f"{name} ({len(new_state)}) {icon}"

    # Watchers
    btn.on_click(toggle)
    search_box.param.watch(on_search, 'value')
    cbg.param.watch(on_cbg_change, 'value')
    
    # Watch state_widget changes (external updates)
    def on_state_change(event):
        # Update button label
        icon = "▲" if container.visible else "▼"
        btn.name = f"{name} ({len(event.new)}) {icon}"
        
        # Update cbg value if needed
        current_visible_opts = cbg.options
        visible_values = list(current_visible_opts.values()) if isinstance(current_visible_opts, dict) else current_visible_opts
        
        expected_cbg_val = [v for v in event.new if v in visible_values]
        
        # Only update if different to avoid loops
        if set(cbg.value) != set(expected_cbg_val):
            nonlocal is_updating_cbg
            is_updating_cbg = True
            cbg.value = expected_cbg_val
            is_updating_cbg = False
            
    state_widget.param.watch(on_state_change, 'value')
    
    # Watch options changes on state_widget
    def on_options_change(event):
        nonlocal opts_dict
        new_options = event.new
        opts_dict = get_opts_dict(new_options)
        
        # Re-apply search filter (which updates cbg.options)
        on_search(type('obj', (object,), {'new': search_box.value})())
        
    state_widget.param.watch(on_options_change, 'options')

    # Layout
    widget_layout = pn.Column(pn.pane.Markdown(f"**{name}**"), btn, container, sizing_mode='stretch_width')
    
    return widget_layout, state_widget

# Helper for Single Select Dropdown
def create_single_select_dropdown(name, options, value):
    # Ensure value is valid
    if value is None and options:
        # Pick first option
        if isinstance(options, list):
            value = options[0]
        elif isinstance(options, dict):
            value = list(options.values())[0]
    
    # 1. The "Source of Truth" widget (hidden)
    state_widget = pn.widgets.Select(name='state', options=options, value=value, visible=False)
    
    # 2. The UI widgets
    search_box = pn.widgets.TextInput(placeholder="Keresés...", sizing_mode='stretch_width', margin=(0, 0, 5, 0))
    
    # Helper to normalize options for filtering
    def get_opts_dict(opts):
        if isinstance(opts, list):
            return {str(v): v for v in opts}
        return opts

    opts_dict = get_opts_dict(options)
        
    # Initial RadioBoxGroup options (all)
    rbg = pn.widgets.RadioBoxGroup(name='', options=options, value=value, sizing_mode='stretch_width')
    
    # Toggle Button
    btn = pn.widgets.Button(name=f"{name}: {value} ▼", button_type='default', css_classes=['dropdown-btn'])
    
    # Container
    container = pn.Column(search_box, rbg, visible=False, max_height=300, scroll=True, css_classes=['dropdown-box'], sizing_mode='stretch_width')
    
    # Logic flags
    is_updating_rbg = False
    
    def toggle(event):
        container.visible = not container.visible
        icon = "▲" if container.visible else "▼"
        btn.name = f"{name}: {state_widget.value} {icon}"
        
    def on_search(event):
        nonlocal is_updating_rbg
        term = event.new.lower() if event.new else ""
        
        # Filter options
        if not term:
            new_opts = state_widget.options # Restore full options from state
        else:
            # Filter keys in opts_dict
            new_opts = {k: v for k, v in opts_dict.items() if term in str(k).lower()}
        
        is_updating_rbg = True
        rbg.options = new_opts
        
        # Update rbg value to match state_widget but only for visible options
        allowed_values = list(new_opts.values()) if isinstance(new_opts, dict) else new_opts
        
        if state_widget.value in allowed_values:
            rbg.value = state_widget.value
        
        is_updating_rbg = False
        
    def on_rbg_change(event):
        nonlocal is_updating_rbg
        if is_updating_rbg: return
        
        # User clicked a radio button -> Update state_widget.value
        new_val = event.new
        
        # Update state widget
        state_widget.value = new_val
        
        # Update button label and close
        btn.name = f"{name}: {new_val} ▼"
        container.visible = False

    # Watchers
    btn.on_click(toggle)
    search_box.param.watch(on_search, 'value')
    rbg.param.watch(on_rbg_change, 'value')
    
    # Watch state_widget changes (external updates)
    def on_state_change(event):
        # Update button label
        icon = "▲" if container.visible else "▼"
        btn.name = f"{name}: {event.new} {icon}"
        
        # Update rbg value if needed
        current_visible_opts = rbg.options
        visible_values = list(current_visible_opts.values()) if isinstance(current_visible_opts, dict) else current_visible_opts
        
        if event.new in visible_values:
            nonlocal is_updating_rbg
            is_updating_rbg = True
            rbg.value = event.new
            is_updating_rbg = False
            
    state_widget.param.watch(on_state_change, 'value')
    
    # Watch options changes on state_widget
    def on_options_change(event):
        nonlocal opts_dict
        new_options = event.new
        opts_dict = get_opts_dict(new_options)
        
        # Re-apply search filter (which updates rbg.options)
        on_search(type('obj', (object,), {'new': search_box.value})())
        
    state_widget.param.watch(on_options_change, 'options')

    # Layout
    widget_layout = pn.Column(pn.pane.Markdown(f"**{name}**"), btn, container, sizing_mode='stretch_width')
    
    return widget_layout, state_widget

# Global Year Selector
year_layout, year_selector = create_dropdown_checkbox('📅 Szezon Kiválasztása', [y for y in range(2021, 2025)], [DEFAULT_YEAR])

# --- Widgets for each tab ---

# 1. Gumistratégiák
tire_race_layout, tire_race_select = create_single_select_dropdown('🏁 Futam', INIT_RACE_OPTIONS, INIT_RACE_VALUES[0] if INIT_RACE_VALUES else None)

# 2. Kvalifikáció
qual_team_layout, qual_team_select = create_dropdown_checkbox('🏎️ Csapat', INIT_TEAM_OPTIONS, [])
qual_driver_layout, qual_driver_select = create_dropdown_checkbox('👤 Versenyző', INIT_DRIVER_OPTIONS, [])
qual_race_layout, qual_race_select = create_dropdown_checkbox('🏁 Futamok', INIT_RACE_OPTIONS, INIT_RACE_VALUES)

# 3. Verseny
race_team_layout, race_team_select = create_dropdown_checkbox('🏎️ Csapat', INIT_TEAM_OPTIONS, [])
race_driver_layout, race_driver_select = create_dropdown_checkbox('👤 Versenyző', INIT_DRIVER_OPTIONS, [])
race_race_layout, race_race_select = create_dropdown_checkbox('🏁 Futamok', INIT_RACE_OPTIONS, INIT_RACE_VALUES)

# 4. Bajnokság
champ_team_layout, champ_team_select = create_dropdown_checkbox('🏎️ Csapat', INIT_TEAM_OPTIONS, [])
champ_driver_layout, champ_driver_select = create_dropdown_checkbox('👤 Versenyző', INIT_DRIVER_OPTIONS, [])

# 5. Pozíció Változás
gain_team_layout, gain_team_select = create_dropdown_checkbox('🏎️ Csapat', INIT_TEAM_OPTIONS, [])
gain_driver_layout, gain_driver_select = create_dropdown_checkbox('👤 Versenyző', INIT_DRIVER_OPTIONS, [])
gain_race_layout, gain_race_select = create_dropdown_checkbox('🏁 Futamok', INIT_RACE_OPTIONS, INIT_RACE_VALUES)

# 6. Köridők
lap_race_layout, lap_race_select = create_dropdown_checkbox('🏁 Futam', INIT_RACE_OPTIONS, [INIT_RACE_VALUES[0]] if INIT_RACE_VALUES else [])
lap_team_layout, lap_team_select = create_dropdown_checkbox('🏎️ Csapat', INIT_TEAM_OPTIONS, [])
lap_driver_layout, lap_driver_select = create_dropdown_checkbox('👤 Versenyző', INIT_DRIVER_OPTIONS, [])

# 7. Map
map_race_layout, map_race_select = create_single_select_dropdown('🏁 Futam (Kördiagram)', INIT_RACE_OPTIONS, INIT_RACE_VALUES[0] if INIT_RACE_VALUES else None)

# 8. Compare
# Left
comp_left_choice_layout, comp_left_choice = create_single_select_dropdown("📊 Diagram Típus", ['Gumistratégiák', 'Kvalifikáció', 'Verseny', 'Bajnokság', 'Pozíció Változás', 'Köridők', 'Térkép'], 'Bajnokság')
comp_left_year_layout, comp_left_year = create_dropdown_checkbox('📅 Év', [y for y in range(2021, 2025)], [DEFAULT_YEAR])
comp_left_race_layout, comp_left_race = create_dropdown_checkbox('🏁 Futam', INIT_RACE_OPTIONS, INIT_RACE_VALUES)
comp_left_team_layout, comp_left_team = create_dropdown_checkbox('🏎️ Csapat', INIT_TEAM_OPTIONS, [])
comp_left_driver_layout, comp_left_driver = create_dropdown_checkbox('👤 Versenyző', INIT_DRIVER_OPTIONS, [])

# Right
comp_right_choice_layout, comp_right_choice = create_single_select_dropdown("📊 Diagram Típus", ['Gumistratégiák', 'Kvalifikáció', 'Verseny', 'Bajnokság', 'Pozíció Változás', 'Köridők', 'Térkép'], 'Verseny')
comp_right_year_layout, comp_right_year = create_dropdown_checkbox('📅 Év', [y for y in range(2021, 2025)], [DEFAULT_YEAR])
comp_right_race_layout, comp_right_race = create_dropdown_checkbox('🏁 Futam', INIT_RACE_OPTIONS, INIT_RACE_VALUES)
comp_right_team_layout, comp_right_team = create_dropdown_checkbox('🏎️ Csapat', INIT_TEAM_OPTIONS, [])
comp_right_driver_layout, comp_right_driver = create_dropdown_checkbox('👤 Versenyző', INIT_DRIVER_OPTIONS, [])


# --- Update Logic ---

def update_widgets_options(years, *widgets_groups):
    """Updates options for a list of widget groups based on selected years."""
    if not years: return
    race_opts, driver_opts, team_opts = get_options(years)
    race_vals = list(race_opts.values())
    default_race = race_vals[0] if race_vals else None
    
    for group in widgets_groups:
        # Unpack group: (race_widget, team_widget, driver_widget) - some might be None
        race_w, team_w, driver_w = group
        
        if race_w:
            race_w.options = race_opts
            # Preserve selection if possible, else reset
            if isinstance(race_w, pn.widgets.Select):
                if race_w.value not in race_vals:
                    race_w.value = default_race
            elif isinstance(race_w, (pn.widgets.MultiChoice, pn.widgets.CheckBoxGroup)):
                new_vals = [v for v in race_w.value if v in race_vals]
                if not new_vals and race_vals:
                    new_vals = race_vals # Select all by default or logic? Gradio kept existing.
                race_w.value = new_vals

        if team_w:
            team_w.options = team_opts
            new_vals = [v for v in team_w.value if v in team_opts]
            team_w.value = new_vals
            
        if driver_w:
            driver_w.options = driver_opts
            new_vals = [v for v in driver_w.value if v in list(driver_opts.values())]
            driver_w.value = new_vals

def on_year_change(event):
    years = event.new
    # Update main tabs widgets
    update_widgets_options(years, 
        (tire_race_select, None, None),
        (qual_race_select, qual_team_select, qual_driver_select),
        (race_race_select, race_team_select, race_driver_select),
        (None, champ_team_select, champ_driver_select),
        (gain_race_select, gain_team_select, gain_driver_select),
        (lap_race_select, lap_team_select, lap_driver_select),
        (map_race_select, None, None)
    )

year_selector.param.watch(on_year_change, 'value')

# Compare tab has its own year selectors
def on_comp_left_year_change(event):
    update_widgets_options(event.new, (comp_left_race, comp_left_team, comp_left_driver))

def on_comp_right_year_change(event):
    update_widgets_options(event.new, (comp_right_race, comp_right_team, comp_right_driver))

comp_left_year.param.watch(on_comp_left_year_change, 'value')
comp_right_year.param.watch(on_comp_right_year_change, 'value')


# --- Plot Bindings ---

# 1. Tire
tire_plot = pn.bind(create_tire_graph, tire_race_select, year_selector)

# 2. Qual
qual_plot = pn.bind(create_qual_graph, qual_driver_select, qual_race_select, qual_team_select, year_selector)

# 3. Race
race_plot = pn.bind(create_race_graph, race_driver_select, race_race_select, race_team_select, year_selector)

# 4. Champ
champ_plot = pn.bind(create_champ_graph, champ_driver_select, champ_team_select, year_selector)

# 5. Gain
gain_plot = pn.bind(create_gain_loss_graph, gain_driver_select, gain_race_select, gain_team_select, year_selector)

# 6. Lap
lap_plot = pn.bind(create_lap_dist_graph, lap_race_select, lap_driver_select, lap_team_select, year_selector)

# 7. Map
map_plot = pn.bind(create_map_graph, year_selector, map_race_select)

# Map Pane with Click Listener
map_pane = pn.pane.plot.Folium(map_plot, sizing_mode='stretch_width', height=450)

def on_hash_change(event):
    if not event.new: return
    hash_val = event.new
    
    # Helper to extract value
    def extract_val(h, key):
        if key + '=' in h:
            try:
                val = h.split(key + '=')[1].split('&')[0]
                return val
            except: return None
        return None

    val = extract_val(hash_val, 'race')
    if val: map_race_select.value = [val]

    val_left = extract_val(hash_val, 'race_left')
    if val_left: comp_left_race.value = [val_left]

    val_right = extract_val(hash_val, 'race_right')
    if val_right: comp_right_race.value = [val_right]

# Watch for URL hash changes to handle map popup clicks
def hook_location():
    if pn.state.location:
        pn.state.location.param.watch(on_hash_change, 'hash')

pn.state.onload(hook_location)

map_pie_plot = pn.bind(create_tire_distribution_chart, map_race_select, year_selector)

# 8. Compare
comp_left_plot = pn.bind(render_selected_plot, comp_left_choice, comp_left_driver, comp_left_race, comp_left_team, comp_left_year, side='left')
comp_right_plot = pn.bind(render_selected_plot, comp_right_choice, comp_right_driver, comp_right_race, comp_right_team, comp_right_year, side='right')


# --- Layout Construction ---

# Helper for tab content layout with "Card" look
def create_tab_layout(widgets, plot, title="Diagram"):
    return pn.Column(
        pn.Row(
            pn.Column(
                pn.pane.Markdown(f"### ⚙️ Beállítások"),
                *widgets,
                css_classes=['widget-box'],
                sizing_mode='stretch_width'
            ),
            sizing_mode='stretch_width'
        ),
        pn.Column(
            pn.pane.Markdown(f"### 📈 {title}"),
            pn.pane.Plotly(plot, sizing_mode='stretch_width', height=500),
            css_classes=['card-box'],
            sizing_mode='stretch_width'
        ),
        sizing_mode='stretch_width',
        margin=(10, 20)
    )

# 1. Tire Tab
tab_tire = create_tab_layout([tire_race_layout], tire_plot, "Gumistratégiák")

# 2. Qual Tab
tab_qual = create_tab_layout([qual_team_layout, qual_driver_layout, qual_race_layout], qual_plot, "Kvalifikációs Eredmények")

# 3. Race Tab
tab_race = create_tab_layout([race_team_layout, race_driver_layout, race_race_layout], race_plot, "Verseny Eredmények")

# 4. Champ Tab
tab_champ = create_tab_layout([champ_team_layout, champ_driver_layout], champ_plot, "Bajnoki Pontverseny")

# 5. Gain Tab
tab_gain = create_tab_layout([gain_team_layout, gain_driver_layout, gain_race_layout], gain_plot, "Pozíció Változás")

# 6. Lap Tab
tab_lap = create_tab_layout([lap_race_layout, lap_team_layout, lap_driver_layout], lap_plot, "Köridők Eloszlása")

# 7. Map Tab
tab_map = pn.Row(
    pn.Column(
        pn.pane.Markdown("### 🗺️ Versenynaptár (Kattints a pontokra!)"),
        map_pane,
        css_classes=['card-box'],
        sizing_mode='stretch_both', # Map takes available space
        min_width=600
    ),
    pn.Column(
        pn.pane.Markdown("### 🍩 Gumihasználat"),
        map_race_layout, # Keep the dropdown too
        pn.pane.Plotly(map_pie_plot, sizing_mode='stretch_width', height=450),
        css_classes=['card-box'],
        sizing_mode='fixed',
        width=450,
        margin=(0, 0, 0, 20)
    ),
    sizing_mode='stretch_both'
)

# 8. Compare Tab
tab_compare = pn.Row(
    pn.Column(
        pn.pane.Markdown("### 👈 Bal Oldal"),
        pn.Column(
            comp_left_choice_layout, comp_left_year_layout, comp_left_race_layout, comp_left_team_layout, comp_left_driver_layout,
            css_classes=['widget-box']
        ),
        pn.Column(
            pn.panel(comp_left_plot, sizing_mode='stretch_width', height=450),
            css_classes=['card-box']
        ),
        sizing_mode='stretch_width', margin=10
    ),
    pn.Column(
        pn.pane.Markdown("### 👉 Jobb Oldal"),
        pn.Column(
            comp_right_choice_layout, comp_right_year_layout, comp_right_race_layout, comp_right_team_layout, comp_right_driver_layout,
            css_classes=['widget-box']
        ),
        pn.Column(
            pn.panel(comp_right_plot, sizing_mode='stretch_width', height=450),
            css_classes=['card-box']
        ),
        sizing_mode='stretch_width', margin=10
    )
)

# Main Tabs
tabs = pn.Tabs(
    ('🛞 Gumik', tab_tire),
    ('⏱️ Kvalifikáció', tab_qual),
    ('🏁 Verseny', tab_race),
    ('🏆 Bajnokság', tab_champ),
    ('📊 Pozíciók', tab_gain),
    ('🏎️ Köridők', tab_lap),
    ('🗺️ Térkép', tab_map),
    ('⚖️ Összehasonlítás', tab_compare),
    dynamic=True
)

# App Template
template = pn.template.FastListTemplate(
    title='F1 Data Viz 2025',
    sidebar=[
        pn.pane.Markdown("## 🌍 Globális Beállítások"),
        year_layout,
        pn.pane.Markdown("---"),
        pn.pane.Markdown("### ℹ️ Info"),
        pn.pane.Markdown("Válassz évet a globális fülekhez. A 'Összehasonlítás' fülön külön évválasztó van.")
    ],
    main=[tabs],
    accent_base_color="#FF1801",
    header_background="#1a1a1a",
    background_color="#121212",
    theme='dark',
    theme_toggle=False,
    shadow=False
)

if __name__.startswith("bokeh"):
    template.servable()
else:
    # For running as a script
    template.show()
