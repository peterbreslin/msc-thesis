import os
import numpy as np


#-----------------------------------------------------------------------------------------------------------------------


def find_starlink_tles(directory):

    # Search for pickle files
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".p"):
            files.append(os.path.join(directory, filename))

    passages = pd.read_pickle(files[0])
    passed_sats = pd.read_pickle(files[1])

    # Load TLEs for all passages
    with open("/net/mascara0/data3/stuik/LaPalma/inputdata/satellites/20221023/3leComplete.txt") as f:
        all_tles = f.readlines()
        f.close()

    # Split TLE list into individual lists for each TLE
    all_tles = [i.strip() for i in all_tles]
    tles = [all_tles[x:x+3] for x in range(0, len(all_tles), 3)]

    # Reduce TLEs to Starlink only
    starlink_tles = []
    for tle in tles:
        if "STARLINK" in tle[0]:
            starlink_tles.append(tle)

    # Obtain satellite passes
    keys = list(passed_sats)

    # Find any Starlink TLEs in the passes
    idx = []
    starlinks = np.asarray(starlink_tles).flatten()
    for key in keys:
        mascara_tle1 = passed_sats[key]['TLE line1'].strip()
        i = np.where(starlinks == mascara_tle1)[0] #this is not going to be fast for big lists...
        if i.size > 0:
            idx.append(i[0] - 1) #appending the name of the starlink sat
            
    # Now have indices for the flattened Starlink TLE list --> divide by 3 to get indices for the original list
    orig_idx = [int(x/3) for x in idx]
    slk_mas_tles = res_list = [starlink_tles[i] for i in orig_idx]

    # Remove 0 labeling of first line of TLE because that's the proper format
    for tle in slk_mas_tles:
        tle[0] = tle[0][2:]

    print(f'Number of satellites recorded for this day: {len(all_tles)}')
    print(f'Number of them that were Starlinks: {len(starlink_tles)}')
    print(f'Number of Starlinks that passed MASCARA: {len(slk_mas_tles)}')

    return slk_mas_tles


#-----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    find_starlink_tles()