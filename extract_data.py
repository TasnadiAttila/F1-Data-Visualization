import fastf1
import pandas as pd
import os

# Cache beállítása
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

def extract_season_data(year=2024):
    print(f"Fetching schedule for {year}...")
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    
    all_stints = []
    all_results = []
    all_laps = []
    all_weather = []
    
    # Csak a versenyeket dolgozzuk fel
    for i, row in schedule.iterrows():
        round_num = row['RoundNumber']
        event_name = row['EventName']
        
        print(f"Processing Round {round_num}: {event_name}...")
        
        try:
            session = fastf1.get_session(year, round_num, 'R')
            # Betöltés telemetria nélkül
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            
            laps = session.laps
            drivers = session.drivers
            
            # Időjárás kezelése (régebbi fastf1 verziókban lehet, hogy nincs weather attribútum)
            try:
                weather = session.weather_data
            except AttributeError:
                try:
                    weather = session.weather
                except AttributeError:
                    weather = pd.DataFrame() # Üres DataFrame ha nincs adat

            
            # --- KVALIFIKÁCIÓS ADATOK BETÖLTÉSE (Q1, Q2, Q3 idők) ---
            # A verseny session results táblája gyakran üresen hagyja a Q időket,
            # ezért külön betöltjük a kvalifikációt, hogy ezeket az adatokat pótoljuk.
            q_times = None
            try:
                session_q = fastf1.get_session(year, round_num, 'Q')
                session_q.load(laps=False, telemetry=False, weather=False, messages=False)
                if hasattr(session_q, 'results'):
                    # Csak a Q időket és az azonosítót vesszük ki
                    q_cols = ['Abbreviation', 'Q1', 'Q2', 'Q3']
                    available_q_cols = [c for c in q_cols if c in session_q.results.columns]
                    if 'Abbreviation' in available_q_cols:
                        q_times = session_q.results[available_q_cols].copy()
            except Exception as e:
                print(f"  -> Info: Kvalifikációs adatok nem elérhetők vagy hiba történt: {e}")

            # --- EREDMÉNYEK ---
            try:
                results = session.results
                # Pozíciók a diagramhoz
                driver_positions = {row['Abbreviation']: row['Position'] for _, row in results.iterrows()}
                
                # Teljes results tábla mentése
                results_df = results.copy()
                
                # Ha sikerült betölteni a valódi Q időket, cseréljük le a Race session (valószínűleg üres) adatait
                if q_times is not None:
                    # Eldobjuk a meglévő (vélhetően NaT) Q oszlopokat
                    results_df = results_df.drop(columns=['Q1', 'Q2', 'Q3'], errors='ignore')
                    # Beolvasztjuk a valódi időket
                    results_df = results_df.merge(q_times, on='Abbreviation', how='left')

                results_df.insert(0, 'EventName', event_name)
                results_df.insert(0, 'RoundNumber', round_num)
                
                # Timedelta konverzió stringre (Excel kompatibilitás)
                for col in results_df.select_dtypes(include=['timedelta64[ns]']).columns:
                    results_df[col] = results_df[col].astype(str)
                
                all_results.append(results_df)
                
            except Exception as e:
                print(f"  -> Warning processing results: {e}")
                driver_positions = {}

            # --- IDŐJÁRÁS ---
            if not weather.empty:
                weather_df = weather.copy()
                weather_df.insert(0, 'EventName', event_name)
                weather_df.insert(0, 'RoundNumber', round_num)
                
                # Timedelta konverzió
                for col in weather_df.select_dtypes(include=['timedelta64[ns]']).columns:
                    weather_df[col] = weather_df[col].astype(str)
                    
                all_weather.append(weather_df)

            # --- KÖRADATOK ---
            laps_df = laps.copy()
            laps_df.insert(0, 'EventName', event_name)
            laps_df.insert(0, 'RoundNumber', round_num)
            
            # NEM törlünk oszlopokat, mindent megtartunk.
            # Csak a típusokat konvertáljuk, hogy az Excel ne dobjon hibát.
            for col in laps_df.select_dtypes(include=['timedelta64[ns]']).columns:
                # Az időket stringként mentjük, így megmarad a pontosság és olvashatóság
                laps_df[col] = laps_df[col].astype(str)
            
            all_laps.append(laps_df)

            # --- STINTEK (Diagramhoz) ---
            for driver in drivers:
                driver_laps = laps.pick_driver(driver)
                if driver_laps.empty:
                    continue
                
                for stint_num, stint_laps in driver_laps.groupby('Stint'):
                    if stint_laps.empty:
                        continue
                        
                    compound = stint_laps['Compound'].iloc[0]
                    if pd.isna(compound) or compound == '':
                        compound = 'UNKNOWN'
                    
                    start_lap = stint_laps['LapNumber'].min()
                    end_lap = stint_laps['LapNumber'].max()
                    duration = end_lap - start_lap + 1
                    
                    all_stints.append({
                        'RoundNumber': round_num,
                        'EventName': event_name,
                        'Driver': driver,
                        'Position': driver_positions.get(driver, 99),
                        'Stint': stint_num,
                        'Compound': compound,
                        'StartLap': start_lap,
                        'EndLap': end_lap,
                        'Duration': duration
                    })
            
            print(f"  -> OK ({len(drivers)} drivers processed)")
            
        except Exception as e:
            print(f"  -> Error processing {event_name}: {e}")
            continue

    # Mentés Excel-be
    output_file = f'{year}_full_season_data.xlsx'
    print(f"\nAdatok mentése Excel fájlba (MINDEN ADAT): {output_file} ...")
    
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            if all_stints:
                pd.DataFrame(all_stints).to_excel(writer, sheet_name='Stints_Summary', index=False)
                print(f"  -> Stints_Summary lap mentve")
            
            if all_results:
                pd.concat(all_results).to_excel(writer, sheet_name='Results_Full', index=False)
                print(f"  -> Results_Full lap mentve")
                
            if all_laps:
                pd.concat(all_laps).to_excel(writer, sheet_name='Laps_Full', index=False)
                print(f"  -> Laps_Full lap mentve")

            if all_weather:
                pd.concat(all_weather).to_excel(writer, sheet_name='Weather_Full', index=False)
                print(f"  -> Weather_Full lap mentve")
                
        print(f"\nSikeres mentés: {output_file}")
        
    except Exception as e:
        print(f"Hiba a mentés során: {e}")

if __name__ == "__main__":
    extract_season_data()
