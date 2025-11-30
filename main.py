import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import fastf1
import fastf1.plotting
from functools import lru_cache

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
    
    all_race_options = []
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
                all_race_options.append({'label': label, 'value': value})
        
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
    team_options = [t for t in sorted(list(teams_set))]
    # Gradiohoz a race_values/labels a legtisztább, ha nem használunk custom components-et
    driver_options_list_of_dict = [{'label': driver_labels.get(d, d), 'value': d} for d in sorted(list(drivers_set))]
    driver_values = [d['value'] for d in driver_options_list_of_dict]
            
    return all_race_options, driver_values, team_options

# Kezdeti adatok
DEFAULT_YEAR = 2022
INIT_RACE_OPTIONS, INIT_DRIVER_OPTIONS, INIT_TEAM_OPTIONS = get_options([DEFAULT_YEAR])
INIT_RACE_VALUES = [r['value'] for r in INIT_RACE_OPTIONS]


# --- Grafikonokat Generáló Függvények ---

def extract_race_params(race_value_list):
    params = []
    for r_val in race_value_list:
        try:
            year, round_number = map(int, str(r_val).split('_'))
            params.append((year, round_number))
        except ValueError:
            continue
    return params

def create_tire_graph(race_value, year_selector_value):
    """Létrehozza a gumistratégia Plotly ábrát."""
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
            
    df_stints['DriverShort'] = (
        df_stints['Driver'].astype(str)
        .str.split()
        .str[-1]
        .str[:3]
        .str.upper()
        .fillna(df_stints['Driver'].astype(str).str[:3].str.upper())
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
        # Use fixed inner plot height so container sizing is consistent in Gradio
        height=450,
        margin=dict(l=120, r=40, t=80, b=80)
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

    # Prepare round->event mapping for shared x-axis labels
    round_label_map = filtered_df.groupby('RoundNumber')['EventName'].first().to_dict()

    filtered_df['Year'] = filtered_df['Year'].astype(str)

    # If single driver selected, draw lines per Year; otherwise color by driver and dash by Year
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
    fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=120, r=40, t=80, b=80))

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

    # Prepare round->event mapping for shared x-axis labels
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
    fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=120, r=40, t=80, b=80))

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
        # keep original abbreviations intact; we'll use 'Year' for coloring
        # and keep the event names separate for tick labels/hover
            
        all_df.append(df)
        
    if not all_df: return go.Figure().update_layout(title="Nincs elérhető adat.", template='plotly_dark')
        
    df_final = pd.concat(all_df)
    df_final = df_final.sort_values(['Year', 'RoundNumber'])

    # Ensure Year is string for Plotly color grouping
    df_final['Year'] = df_final['Year'].astype(str)

    hover_data = {'Points': True, 'Position': True, 'Abbreviation': True, 'Year': True}

    # Prepare x-axis ticks (one shared set). Use RoundNumber as x, map to an EventName label.
    round_label_map = df_final.groupby('RoundNumber')['EventName'].first().to_dict()

    # If a single driver is selected (or filtering by a team resulting in one driver),
    # draw lines per Year so the same driver across years produces separate lines.
    if selected_drivers and len(selected_drivers) == 1:
        fig = px.line(
            df_final, x='RoundNumber', y='CumulativePoints', color='Year', markers=True,
            title="Világbajnoki Pontverseny Alakulása", hover_data=hover_data
        )
    else:
        # Multiple drivers: color by driver and differentiate years with line dashes
        fig = px.line(
            df_final, x='RoundNumber', y='CumulativePoints', color='Abbreviation', line_dash='Year', markers=True,
            title="Világbajnoki Pontverseny Alakulása", hover_data=hover_data
        )

    # Apply single x-axis labels using the round->event mapping
    tick_vals = sorted(round_label_map.keys())
    tick_texts = [round_label_map[r] for r in tick_vals]
    fig.update_xaxes(tickmode='array', tickvals=tick_vals, ticktext=tick_texts)
    
    fig.update_yaxes(title="Összpontszám")
    fig.update_xaxes(title="Futam")
    fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=120, r=40, t=80, b=80))
    
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

    # Horizontal bars: x shows the position change, y shows the event (single shared axis)
    fig.update_xaxes(title="Pozíció Változás (+ Javított, - Rontott)")
    fig.update_yaxes(title="Futam", automargin=True)
    fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=220, r=40, t=80, b=80))
    
    return fig


def render_selected_plot(choice, selected_drivers, selected_races, selected_teams, years):
    """Generic renderer for the compare view. Chooses the right plotting function.

    Parameters mirror the compare UI: `selected_races` may be a list; for plots
    that expect a single race, the first element will be used.
    """
    if not choice:
        return go.Figure().update_layout(title="Válassz diagram típust", template='plotly_dark')

    # Normalize years
    if not isinstance(years, list):
        years = [years] if years is not None else [DEFAULT_YEAR]

    # Ensure inputs are in the expected form
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
        else:
            return go.Figure().update_layout(title="Ismeretlen diagram típus.", template='plotly_dark')
    except Exception as e:
        return go.Figure().update_layout(title=f"Hiba a diagram létrehozásakor: {e}", template='plotly_dark')

def create_lap_dist_graph(race_value, selected_drivers, selected_teams, years):
    """Létrehozza a köridő eloszlás Plotly ábráját."""
    if not race_value:
        return go.Figure().update_layout(title="Válassz egy futamot!", template='plotly_dark')
    
    try:
        year, round_number = map(int, str(race_value).split('_'))
    except ValueError:
        return go.Figure().update_layout(title="Érvénytelen futam kiválasztás.", template='plotly_dark')

    try:
        # FastF1 adatbetöltés szükséges itt, mert a .xlsx-ben nincs köridő adat
        session = fastf1.get_session(year, round_number, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        
        laps = laps.pick_wo_box()
        
        # A fastf1-ben a 'Driver' oszlop tartalmazza az abbreviációt
        if selected_drivers:
            laps = laps[laps['Driver'].isin(selected_drivers)]
        elif selected_teams:
            laps = laps[laps['Team'].isin(selected_teams)]
        
        if laps.empty:
            return go.Figure().update_layout(title="Nincs elérhető köridő adat a kiválasztott szűréshez.", template='plotly_dark')
        
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        
        # Egyszerű szűrés a skála torzítás ellen
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
        fig.update_layout(template='plotly_dark', autosize=True, height=450, margin=dict(l=120, r=40, t=80, b=80))
        
        return fig

    except Exception as e:
        return go.Figure().update_layout(title=f"Hiba az adatok betöltésekor: {str(e)}", template='plotly_dark')

# --- Gradio UI és Callback Logika ---

def update_all_options(years):
    """
    Frissíti az összes Dropdown opcióját az évválasztó változásakor.
    """
    # If no year is provided (e.g. user cleared the selector), do NOT overwrite
    # the existing dropdown choices/values in the UI. Returning gr.update()
    # objects with no args tells Gradio to leave those components unchanged.
    if not years:
        return tuple([gr.update() for _ in range(17)])

    # get_options a Gradio-nak megfelelő formátumban adja vissza a listákat
    race_options, driver_values, team_values = get_options(years)
    
    # A Gradio 4+ Dropdown a values listáját is tudja fogadni choices-ként
    race_values = [r['value'] for r in race_options]
    default_race_val = race_values[0] if race_values else None

    # Build per-component updates so we only change the components we intend to.
    updates = [
        gr.update(choices=race_values),            # tire_race_dropdown choices
        gr.update(value=default_race_val),         # tire_race_dropdown value
        gr.update(choices=race_values),            # qual_race_dropdown choices
        gr.update(choices=race_values),            # race_race_dropdown choices
        gr.update(choices=race_values),            # gain_race_dropdown choices
        gr.update(choices=race_values),            # lap_race_dropdown choices
        gr.update(value=default_race_val),         # lap_race_dropdown value
        gr.update(choices=team_values),            # qual_team_dropdown choices
        gr.update(choices=driver_values),          # qual_driver_dropdown choices
        gr.update(choices=team_values),            # race_team_dropdown choices
        gr.update(choices=driver_values),          # race_driver_dropdown choices
        gr.update(choices=team_values),            # champ_team_dropdown choices
        gr.update(choices=driver_values),          # champ_driver_dropdown choices
        gr.update(choices=team_values),            # gain_team_dropdown choices
        gr.update(choices=driver_values),          # gain_driver_dropdown choices
        gr.update(choices=team_values),            # lap_team_dropdown choices
        gr.update(choices=driver_values)           # lap_driver_dropdown choices
    ]

    return tuple(updates)

# A Gradio Blocks felépítése
# !!! HIBAJAVÍTÁS: ELTÁVOLÍTVA A 'theme' ARGUMENTUM !!!
with gr.Blocks(title="F1 Szezon Elemző") as app:
    gr.Markdown("<h1>F1 Szezon Elemző (Gradio)</h1>")
    # Ensure Plotly plot containers use the full available width and consistent height
    gr.HTML("""
    <style>
    .js-plotly-plot, .plotly-graph-div { width: 100% !important; }
    .js-plotly-plot .svg-container { width: 100% !important; height: 450px !important; }
    </style>
    """)

    # --- Vezérlőpult ---
    with gr.Row():
        with gr.Column(scale=1):
            year_selector = gr.Dropdown(
                label="Év",
                choices=[y for y in range(2021, 2025)],
                value=[DEFAULT_YEAR],
                multiselect=True,
                allow_custom_value=True
            )
        with gr.Column(scale=2):
            gr.Markdown("A Gradio elrendezése fixen fülönkénti.")

    # --- Graph Containers (Fülek) ---
    with gr.Tabs() as tabs:
        
        # 1. Gumistratégiák
        with gr.TabItem("Gumistratégiák", id=0):
            with gr.Row():
                tire_race_dropdown = gr.Dropdown(
                    label="Válassz futamot:",
                    choices=INIT_RACE_VALUES,
                    value=INIT_RACE_VALUES[0] if INIT_RACE_VALUES else None,
                    interactive=True,
                    allow_custom_value=True
                )
            # !!! HIBAJAVÍTÁS: ELTÁVOLÍTVA A 'height' ARGUMENTUM !!!
            with gr.Row():
                tire_graph = gr.Plot(label="Gumistratégiák Ábrája")
            
            # Callback a legördülő menük frissítésére
            year_selector.change(
                fn=update_all_options,
                inputs=[year_selector],
                outputs=[
                    tire_race_dropdown, tire_race_dropdown,
                    gr.State(), gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State()
                ],
                queue=True
            )
            # Callback a grafikon frissítésére
            tire_race_dropdown.change(
                fn=create_tire_graph,
                inputs=[tire_race_dropdown, year_selector],
                outputs=[tire_graph],
                queue=False
            )


        # 2. Kvalifikációs Eredmények
        with gr.TabItem("Kvalifikáció", id=1):
            with gr.Row():
                qual_team_dropdown = gr.Dropdown(label="Válassz csapatot:", choices=INIT_TEAM_OPTIONS, multiselect=True, allow_custom_value=True)
                qual_driver_dropdown = gr.Dropdown(label="Válassz versenyzőket:", choices=INIT_DRIVER_OPTIONS, multiselect=True, allow_custom_value=True)
            with gr.Row():
                qual_race_dropdown = gr.Dropdown(
                    label="Válassz futamokat:",
                    choices=INIT_RACE_VALUES,
                    value=INIT_RACE_VALUES,
                    multiselect=True,
                    allow_custom_value=True
                )
            # !!! HIBAJAVÍTÁS: ELTÁVOLÍTVA A 'height' ARGUMENTUM !!!
            with gr.Row():
                qual_graph = gr.Plot(label="Rajthelyek alakulása")

            # Callback: frissítjük az opciókat (a többi fülnél is ezt használjuk, csak a megfelelő outputokat frissítjük)
            year_selector.change(
                fn=update_all_options,
                inputs=[year_selector],
                outputs=[
                    gr.State(), gr.State(),
                    qual_race_dropdown, 
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    qual_team_dropdown, qual_driver_dropdown, 
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State()
                ],
                queue=True
            )
            # Callback: a grafikon frissítése
            qual_inputs = [qual_driver_dropdown, qual_race_dropdown, qual_team_dropdown, year_selector]
            for input_comp in qual_inputs:
                input_comp.change(
                    fn=create_qual_graph,
                    inputs=qual_inputs,
                    outputs=[qual_graph],
                    queue=False
                )


        # 3. Verseny Eredmények
        with gr.TabItem("Verseny", id=2):
            with gr.Row():
                race_team_dropdown = gr.Dropdown(label="Válassz csapatot:", choices=INIT_TEAM_OPTIONS, multiselect=True, allow_custom_value=True)
                race_driver_dropdown = gr.Dropdown(label="Válassz versenyzőket:", choices=INIT_DRIVER_OPTIONS, multiselect=True, allow_custom_value=True)
            with gr.Row():
                race_race_dropdown = gr.Dropdown(
                    label="Válassz futamokat:",
                    choices=INIT_RACE_VALUES,
                    value=INIT_RACE_VALUES,
                    multiselect=True,
                    allow_custom_value=True
                )
            # !!! HIBAJAVÍTÁS: ELTÁVOLÍTVA A 'height' ARGUMENTUM !!!
            with gr.Row():
                race_graph = gr.Plot(
                    label="Versenyeredmények alakulása",
                    value=go.Figure(layout=go.Layout(template='plotly_dark', width=1200, height=450, margin=dict(l=120, r=40, t=80, b=80)))
                )

            year_selector.change(
                fn=update_all_options,
                inputs=[year_selector],
                outputs=[
                    gr.State(), gr.State(),
                    gr.State(),
                    race_race_dropdown, 
                    gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    race_team_dropdown, race_driver_dropdown, 
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State()
                ],
                queue=True
            )
            race_inputs = [race_driver_dropdown, race_race_dropdown, race_team_dropdown, year_selector]
            for input_comp in race_inputs:
                input_comp.change(
                    fn=create_race_graph,
                    inputs=race_inputs,
                    outputs=[race_graph],
                    queue=False
                )

        # 4. Világbajnoki Pontverseny
        with gr.TabItem("Bajnokság", id=3):
            with gr.Row():
                champ_team_dropdown = gr.Dropdown(label="Válassz csapatot:", choices=INIT_TEAM_OPTIONS, multiselect=True, allow_custom_value=True)
                champ_driver_dropdown = gr.Dropdown(label="Válassz versenyzőket:", choices=INIT_DRIVER_OPTIONS, multiselect=True, allow_custom_value=True)
            # !!! HIBAJAVÍTÁS: ELTÁVOLÍTVA A 'height' ARGUMENTUM !!!
            with gr.Row():
                champ_graph = gr.Plot(
                    label="Világbajnoki Pontverseny Alakulása",
                    value=go.Figure(layout=go.Layout(template='plotly_dark', width=1200, height=450, margin=dict(l=120, r=40, t=80, b=80)))
                )

            year_selector.change(
                fn=update_all_options,
                inputs=[year_selector],
                outputs=[
                    gr.State(), gr.State(),
                    gr.State(), gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    champ_team_dropdown, champ_driver_dropdown, 
                    gr.State(), gr.State(),
                    gr.State(), gr.State()
                ],
                queue=True
            )
            champ_inputs = [champ_driver_dropdown, champ_team_dropdown, year_selector]
            for input_comp in champ_inputs:
                input_comp.change(
                    fn=create_champ_graph,
                    inputs=champ_inputs,
                    outputs=[champ_graph],
                    queue=False
                )


        # 5. Pozíció Változás
        with gr.TabItem("Pozíció Változás", id=4):
            with gr.Row():
                gain_team_dropdown = gr.Dropdown(label="Válassz csapatot:", choices=INIT_TEAM_OPTIONS, multiselect=True, allow_custom_value=True)
                gain_driver_dropdown = gr.Dropdown(label="Válassz versenyzőket:", choices=INIT_DRIVER_OPTIONS, multiselect=True, allow_custom_value=True)
            with gr.Row():
                gain_race_dropdown = gr.Dropdown(
                    label="Válassz futamokat:",
                    choices=INIT_RACE_VALUES,
                    value=INIT_RACE_VALUES,
                    multiselect=True,
                    allow_custom_value=True
                )
            # !!! HIBAJAVÍTÁS: ELTÁVOLÍTVA A 'height' ARGUMENTUM !!!
            with gr.Row():
                gain_graph = gr.Plot(
                    label="Pozíció Nyereség/Veszteség",
                    value=go.Figure(layout=go.Layout(template='plotly_dark', width=1200, height=450, margin=dict(l=120, r=40, t=80, b=80)))
                )

            year_selector.change(
                fn=update_all_options,
                inputs=[year_selector],
                outputs=[
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gain_race_dropdown, 
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gain_team_dropdown, gain_driver_dropdown, 
                    gr.State(), gr.State()
                ],
                queue=True
            )
            gain_inputs = [gain_driver_dropdown, gain_race_dropdown, gain_team_dropdown, year_selector]
            for input_comp in gain_inputs:
                input_comp.change(
                    fn=create_gain_loss_graph,
                    inputs=gain_inputs,
                    outputs=[gain_graph],
                    queue=False
                )

        # 6. Köridők Eloszlása
        with gr.TabItem("Köridők", id=5):
            with gr.Row():
                lap_race_dropdown = gr.Dropdown(
                    label="Válassz futamot:",
                    choices=INIT_RACE_VALUES,
                    value=INIT_RACE_VALUES[0] if INIT_RACE_VALUES else None,
                    interactive=True,
                    allow_custom_value=True
                )
            with gr.Row():
                lap_team_dropdown = gr.Dropdown(label="Válassz csapatot:", choices=INIT_TEAM_OPTIONS, multiselect=True, allow_custom_value=True)
                lap_driver_dropdown = gr.Dropdown(label="Válassz versenyzőket:", choices=INIT_DRIVER_OPTIONS, multiselect=True, allow_custom_value=True)
            # !!! HIBAJAVÍTÁS: ELTÁVOLÍTVA A 'height' ARGUMENTUM !!!
            with gr.Row():
                lap_graph = gr.Plot(label="Köridők Eloszlása Box Plot")

            year_selector.change(
                fn=update_all_options,
                inputs=[year_selector],
                outputs=[
                    gr.State(), gr.State(),
                    gr.State(), gr.State(), gr.State(),
                    lap_race_dropdown, lap_race_dropdown, 
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    gr.State(), gr.State(),
                    lap_team_dropdown, lap_driver_dropdown 
                ],
                queue=True
            )
            lap_inputs = [lap_race_dropdown, lap_driver_dropdown, lap_team_dropdown, year_selector]
            for input_comp in lap_inputs:
                input_comp.change(
                    fn=create_lap_dist_graph,
                    inputs=lap_inputs,
                    outputs=[lap_graph],
                    queue=False
                )

        # 7. Compare (side-by-side)
        with gr.TabItem("Compare", id=6):
            with gr.Row():
                # Left column controls
                with gr.Column():
                    compare_left_choice = gr.Dropdown(label="Bal diagram típusa:", choices=[
                        'Gumistratégiák', 'Kvalifikáció', 'Verseny', 'Bajnokság', 'Pozíció Változás', 'Köridők'
                    ], value='Bajnokság')
                    compare_left_year = gr.Dropdown(label='Év (bal)', choices=[y for y in range(2021, 2025)], value=[DEFAULT_YEAR], multiselect=True, allow_custom_value=True)
                    compare_left_race = gr.Dropdown(label='Futam (bal)', choices=INIT_RACE_VALUES, value=INIT_RACE_VALUES[0] if INIT_RACE_VALUES else None, multiselect=True, allow_custom_value=True)
                    compare_left_team = gr.Dropdown(label='Csapat (bal)', choices=INIT_TEAM_OPTIONS, multiselect=True, allow_custom_value=True)
                    compare_left_driver = gr.Dropdown(label='Versenyző (bal)', choices=INIT_DRIVER_OPTIONS, multiselect=True, allow_custom_value=True)
                    compare_left_update = gr.Button('Frissít (bal)')
                    compare_left_plot = gr.Plot(label='Bal Diagram')

                # Right column controls
                with gr.Column():
                    compare_right_choice = gr.Dropdown(label="Jobb diagram típusa:", choices=[
                        'Gumistratégiák', 'Kvalifikáció', 'Verseny', 'Bajnokság', 'Pozíció Változás', 'Köridők'
                    ], value='Verseny')
                    compare_right_year = gr.Dropdown(label='Év (jobb)', choices=[y for y in range(2021, 2025)], value=[DEFAULT_YEAR], multiselect=True, allow_custom_value=True)
                    compare_right_race = gr.Dropdown(label='Futam (jobb)', choices=INIT_RACE_VALUES, value=INIT_RACE_VALUES[0] if INIT_RACE_VALUES else None, multiselect=True, allow_custom_value=True)
                    compare_right_team = gr.Dropdown(label='Csapat (jobb)', choices=INIT_TEAM_OPTIONS, multiselect=True, allow_custom_value=True)
                    compare_right_driver = gr.Dropdown(label='Versenyző (jobb)', choices=INIT_DRIVER_OPTIONS, multiselect=True, allow_custom_value=True)
                    compare_right_update = gr.Button('Frissít (jobb)')
                    compare_right_plot = gr.Plot(label='Jobb Diagram')

            # Wire buttons to renderer
            compare_left_update.click(
                fn=render_selected_plot,
                inputs=[compare_left_choice, compare_left_driver, compare_left_race, compare_left_team, compare_left_year],
                outputs=[compare_left_plot]
            )
            compare_right_update.click(
                fn=render_selected_plot,
                inputs=[compare_right_choice, compare_right_driver, compare_right_race, compare_right_team, compare_right_year],
                outputs=[compare_right_plot]
            )

            # Auto-update when any control changes (no need to click the button)
            for inp in [compare_left_choice, compare_left_driver, compare_left_race, compare_left_team, compare_left_year]:
                inp.change(
                    fn=render_selected_plot,
                    inputs=[compare_left_choice, compare_left_driver, compare_left_race, compare_left_team, compare_left_year],
                    outputs=[compare_left_plot],
                    queue=False
                )

            for inp in [compare_right_choice, compare_right_driver, compare_right_race, compare_right_team, compare_right_year]:
                inp.change(
                    fn=render_selected_plot,
                    inputs=[compare_right_choice, compare_right_driver, compare_right_race, compare_right_team, compare_right_year],
                    outputs=[compare_right_plot],
                    queue=False
                )


# Move app.load calls inside the main Blocks context (after all UI definitions)
    app.load(
        fn=create_tire_graph,
        inputs=[tire_race_dropdown, year_selector],
        outputs=[tire_graph],
        queue=False
    )
    app.load(
        fn=create_qual_graph,
        inputs=[qual_driver_dropdown, qual_race_dropdown, qual_team_dropdown, year_selector],
        outputs=[qual_graph],
        queue=False
    )
    app.load(
        fn=create_race_graph,
        inputs=[race_driver_dropdown, race_race_dropdown, race_team_dropdown, year_selector],
        outputs=[race_graph],
        queue=False
    )
    app.load(
        fn=create_champ_graph,
        inputs=[champ_driver_dropdown, champ_team_dropdown, year_selector],
        outputs=[champ_graph],
        queue=False
    )
    app.load(
        fn=create_gain_loss_graph,
        inputs=[gain_driver_dropdown, gain_race_dropdown, gain_team_dropdown, year_selector],
        outputs=[gain_graph],
        queue=False
    )
    app.load(
        fn=create_lap_dist_graph,
        inputs=[lap_race_dropdown, lap_driver_dropdown, lap_team_dropdown, year_selector],
        outputs=[lap_graph],
        queue=False
    )

if __name__ == "__main__":
    app.launch()