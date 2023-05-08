import unit_utils
import os

files = {
        'WT': [{
            'rating_file': 'D:\\TetrodeData\\2022-03-23_09-17-15-p\\ClusterRating_2022-03-23_09-17-15-p.csv',
            'directories': ['D:\\TetrodeData\\2022-03-23_09-17-15-p\\Sleep1',
                            'D:\\TetrodeData\\2022-03-23_09-17-15-p\\Sleep2'],
            'tetrodes': list(range(1, 6))
        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-03-24_09-28-38-p\\ClusterRating_2022-03-24_09-28-38-p.csv',
            'directories': ['D:\\TetrodeData\\2022-03-24_09-28-38-p\\Sleep1',
                            'D:\\TetrodeData\\2022-03-24_09-28-38-p\\Sleep2',
                            'D:\\TetrodeData\\2022-03-24_09-28-38-p\\Sleep3'],
            'tetrodes': list(range(1, 6))

        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-07-10_13-27-33-p\\ClusterRating_2022-07-10_13-27-33-p.csv',
            'directories': ['D:\\TetrodeData\\2022-07-10_13-27-33-p\\Sleep1',
                            'D:\\TetrodeData\\2022-07-10_13-27-33-p\\Sleep2'],
            'tetrodes': [2,3,5,6,7,8]
        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-07-11_13-47-33-p\\ClusterRating_2022-07-11_13-47-33-p.csv',
            'directories': ['D:\\TetrodeData\\2022-07-11_13-47-33-p\\Sleep1',
                            'D:\\TetrodeData\\2022-07-11_13-47-33-p\\Sleep2',
                            'D:\\TetrodeData\\2022-07-11_13-47-33-p\\Sleep3'],
            'tetrodes': [2,3,5,6,7,8]
        }
        ],
        'TG': [{
            'rating_file': 'D:\\TetrodeData\\2022-04-12-09-29-11-p\\ClusterRating_2022-04-12_09-29-11-p.csv',
            'directories': ['D:\\TetrodeData\\2022-04-12-09-29-11-p\\Sleep1',
                            'D:\\TetrodeData\\2022-04-12-09-29-11-p\\Sleep2'],
            'tetrodes': list(range(1, 9))
        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-04-13_09-31-41-p\\ClusterRating_2022-04-13_09-31-41-p.csv',
            'directories': ['D:\\TetrodeData\\2022-04-13_09-31-41-p\\Sleep1',
                            'D:\\TetrodeData\\2022-04-13_09-31-41-p\\Sleep2',
                            'D:\\TetrodeData\\2022-04-13_09-31-41-p\\Sleep3'],
            'tetrodes': list(range(1, 9))
        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-07-08_10-01-42-p\\ClusterRating_2022-07-08_10-01-42-p.csv',
            'directories': ['D:\\TetrodeData\\2022-07-08_10-01-42-p\\Sleep1',
                            'D:\\TetrodeData\\2022-07-08_10-01-42-p\\Sleep2'],
            'tetrodes': [1, 2, 3, 5, 7, 8]
        },
        {
            'rating_file': 'D:\\TetrodeData\\2022-07-09_09-54-32-p\\ClusterRating_2022-07-09_09-54-32-p.csv',
            'directories': ['D:\\TetrodeData\\2022-07-09_09-54-32-p\\Sleep1',
                            'D:\\TetrodeData\\2022-07-09_09-54-32-p\\Sleep2',
                            'D:\\TetrodeData\\2022-07-09_09-54-32-p\\Sleep3'],
            'tetrodes': [1, 2, 3, 5, 7, 8]
        }
        ]
    }

data = {}

for strain in files:
    pyr_unit_count = 0
    tg_unit_count = 0

    for k in files[strain]:

        rating_file = k['rating_file']
        directories = k['directories']
        tetrodes = k['tetrodes']

        units = unit_utils.good_units(os.path.dirname(directories[0]), rating_file, tetrodes)
        pyr_units, int_units = unit_utils.split_unit_types(os.path.join(os.path.dirname(rating_file), 'putative_interneurons.csv'), units)

        pyr_unit_count += len(pyr_units)
        tg_unit_count += len(int_units)

    data[strain] = {
        "pyr": pyr_unit_count,
        "int": tg_unit_count
    }

print(data)