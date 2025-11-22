import fastf1
import os
import time

# Ensure cache dir exists and enable fastf1 cache
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

print('Starting fetch of 2021 season into cache...')

schedule = fastf1.get_event_schedule(2021, include_testing=False)

sessions = ['FP1', 'FP2', 'FP3', 'Q', 'R']

# Count total tasks for progress
total_tasks = len(schedule) * len(sessions)
current = 0

for idx, row in schedule.iterrows():
    round_num = int(row['RoundNumber'])
    event_name = row['EventName']
    for ses in sessions:
        current += 1
        print(f"[{current}/{total_tasks}] Checking: {event_name} (Round {round_num}) - {ses}")
        try:
            # If session data already exists in cache, skip downloading
            def _norm(s):
                return ''.join(c for c in str(s).lower() if c.isalnum())

            # map fastf1 session short names to cache folder keywords
            ses_map = {
                'FP1': 'practice_1',
                'FP2': 'practice_2',
                'FP3': 'practice_3',
                'Q': 'qualifying',
                'R': 'race'
            }

            need_download = True
            year_dir = os.path.join('cache', '2021')
            if os.path.isdir(year_dir):
                norm_event = _norm(event_name)
                for event_dir in os.listdir(year_dir):
                    if norm_event in _norm(event_dir):
                        event_path = os.path.join(year_dir, event_dir)
                        # look for a session folder that matches this session
                        for sub in os.listdir(event_path):
                            if ses_map[ses] in sub.lower():
                                print('  -> Found in cache, skipping')
                                need_download = False
                                break
                        if not need_download:
                            break

            if need_download:
                print('  -> Not in cache, downloading...')
                session = fastf1.get_session(2021, round_num, ses)
                session.load()
                print('  -> Downloaded and cached')

        except Exception as e:
            print(f'  -> Failed: {e}')
        # polite pause to avoid overwhelming the server
        time.sleep(1)

print('Finished fetching 2021 season.')
