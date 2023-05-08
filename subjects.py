import os
import scipy.io as sio
import unit_utils
import glob


subjects = {
    '18-1': {
        'habdir': 'D:/Data/03-23-2022',
        'olmdir': 'D:/Data/03-24-2022',
        'tetrodes': [1,2,3,4,5],
        'strain': 'wt',
        'sex': 'male'
    },
    '25-10': {
        'habdir': 'D:/Data/04-12-2022',
        'olmdir': 'D:/Data/04-13-2022',
        'tetrodes': [1,2,3,4,5,6,7,8],
        'strain': 'tg',
        'sex': 'male'
    },
    '36-3': {
        'habdir': 'D:/Data/07-08-2022',
        'olmdir': 'D:/Data/07-09-2022',
        'tetrodes': [1,2,3,5,7,8],
        'strain': 'tg',
        'sex': 'male'
    },
    '36-1': {
        'habdir': 'D:/Data/07-10-2022',
        'olmdir': 'D:/Data/07-11-2022',
        'tetrodes': [2,3,5,6,7,8],
        'strain': 'wt',
        'sex': 'male'
    },
    '64-30': {
        'habdir': 'D:/Data/02-24-2023',
        'olmdir': 'D:/Data/02-25-2023',
        'tetrodes': [2,3,4,5,6,8],
        'strain': 'wt',
        'sex': 'male'
    },
    '59-2': {
        'habdir': 'D:/Data/03-21-2023',
        'olmdir': 'D:/Data/03-22-2023',
        'tetrodes': [3,6,7],
        'strain': 'wt',
        'sex': 'female'
    },
    '62-1': {
        'habdir': 'D:/Data/04-01-2023',
        'olmdir': 'D:/Data/04-02-2023',
        'tetrodes': [1,2,3,4,5,8],
        'strain': 'tg',
        'sex': 'female'
    }
}

sessions = {
    'hab': [
        'Sleep1', 'Sleep2'
    ],
    'olm': [
        'Sleep1', 'Sleep2', 'Sleep3'
    ]

}

# what to load
def load_data(subject, units=False, lfp=False, ripples=False, spindles=False, phases=False):
    data = {
        'hab': {},
        'olm': {}
    }
    s = subject


    for task in sessions:

        pyr_units = None
        int_units = None
        sample_rate = None

        # load data
        if units:
            rating_file = glob.glob(s[task + 'dir'] + '/ClusterRating*.csv')[0]
            units = unit_utils.good_units(s[task + 'dir'], rating_file, s['tetrodes'])
            pyr_units, int_units = unit_utils.split_unit_types(s[task + 'dir'] + '/putative_interneurons.csv', units)




        data[task] = {
            'pyr_units': pyr_units,
            'int_units': int_units,
        }

        for session in sessions[task]:

            lfp_data = sio.loadmat(os.path.join(s[task + 'dir'], session, "data.mat"))
            ripple_data = sio.loadmat(os.path.join(s[task + 'dir'], session, "ripples.mat"))
            ripple_windows = ripple_data['ripples']['windows'][0, 0]
            ts = lfp_data['data']['ts'][0, 0].flatten()
            sample_rate = lfp_data['data']['sampleRate'][0, 0][0, 0]

            ripple_phases = {}
            if phases:
                phase_data = sio.loadmat(os.path.join(s[task + 'dir'], session, "ripple_phases.mat"))
                for tt in phase_data['ripple_phases'].dtype.names:
                    ripple_phases[tt] = phase_data['ripple_phases'][tt][0,0].flatten()

            data[task][session] = {
                'ts': ts,
                'ripple_windows': ripple_windows,
                'ripple_phases': ripple_phases
            }


        data[task]['sample_rate'] = sample_rate


    return data


# count units
"""for subject in subjects:
    data = load_data(subjects[subject], units=True)
    print(subject)
    print(len(data['hab']['pyr_units']))
    print(len(data['hab']['int_units']))
    print(len(data['olm']['pyr_units']))
    print(len(data['olm']['int_units']))"""