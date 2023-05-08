import pandas as pd
import os
import subjects


# return duration of ripples (ms)
def ripple_durations(ripple_windows, sample_rate):
    durations = []
    onsets = []
    for r in ripple_windows.T:
        duration = (r[1] - r[0]) / (sample_rate / 1000)
        durations.append(duration)
        onsets.append(r[0] / sample_rate)

    return durations, onsets


# return number of ripples + ripple density
def ripple_incidence(ripple_windows ,ts, sample_rate):
    start = 0
    stop = len(ts)

    # calculate total number of ripples
    num_ripples = ripple_windows.shape[1]

    # calculate total time
    total_time = stop / sample_rate

    # density ripples/second
    density = num_ripples/total_time

    # calculate num ripples in first 30 mins
    r_s = ripple_windows[0,:]
    ripples_in_window = r_s[(r_s >= start) & (r_s <= start + 30 * 60 * sample_rate)]
    density_30 = len(ripples_in_window) / (30 * 60 * sample_rate)

    # first 1 hour
    ripples_in_window = r_s[(r_s >= start) & (r_s <= start + 60 * 60 * sample_rate)]
    density_60 = len(ripples_in_window) / (60 * 60 * sample_rate)

    return num_ripples, density, density_30, density_60, total_time

def ripple_times():
    pass


# run pipeline
ids = ['18-1', '25-10', '36-1', '36-3', '64-30','59-2', '62-1']
# = ['59-2', '62-1']
for id in ids:
    subject = subjects.subjects[id]

    data = subjects.load_data(subject, units=False)

    for task in subjects.sessions:
        for session in subjects.sessions[task]:
            # task = hab/olm, session = sleep
            print(id + task + session)

            # generate ripple stats for all sleep sessions and save as csv
            # incidence
            num_ripples, density, density_30, density_60, total_time = ripple_incidence(data[task][session]['ripple_windows'], data[task][session]['ts'], data[task]['sample_rate'])
            df = pd.DataFrame({'num_ripples': num_ripples, 'density': density, 'density_30': density_30, 'density_60': density_60, 'id': id, 'strain': subject['strain'], 'task': task, 'session': session, 'total_time': total_time}, index=[0])
            df.to_csv(os.path.join(subject[task + 'dir'], session, 'ripple_incidence.csv'))

            durations, onsets = ripple_durations(data[task][session]['ripple_windows'], data[task]['sample_rate'])
            df = pd.DataFrame({'durations': durations, 'onsets': onsets, 'id': [id] * len(durations), 'strain': [subject['strain']] * len(durations), 'task': [task] * len(durations), 'session': [session] * len(durations)})
            df.to_csv(os.path.join(subject[task + 'dir'], session, 'ripple_durations.csv'))