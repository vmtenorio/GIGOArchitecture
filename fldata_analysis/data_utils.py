import pandas as pd

def remove_airports(df, nflights):
    ndays = (df['FL_DATE'].max() - df['FL_DATE'].min()).days

    airports = pd.concat([df['ORIGIN_AIRPORT_ID'], df['DEST_AIRPORT_ID']]).drop_duplicates()

    airports_cons = []

    for a in airports:
        flights_a = df[(df['ORIGIN_AIRPORT_ID'] == a) | (df['DEST_AIRPORT_ID'] == a)]
        if len(flights_a.index) / ndays > nflights:
            airports_cons.append(a)

    print("Airports considered: " + str(len(airports_cons)))
    #print("Airports considered: " + str(airports_cons))

    # Discuss | vs & here: the two airports must be between the considered airports?
    return df[df['ORIGIN_AIRPORT_ID'].isin(airports_cons) & df['DEST_AIRPORT_ID'].isin(airports_cons)]

def hour_proc(flights_hour, airports):
    dep_del = {}
    arr_del = {}
    # Airport delay at time t as the average of the departure delay of all the flights
    for a in airports:
        flights_ori_a = flights_hour[flights_hour['ORIGIN_AIRPORT_ID'] == a]
        if len(flights_ori_a) > 0:
            dep_del[str(a)] = flights_ori_a['DEP_DELAY'].sum() / len(flights_ori_a)
        else:
            dep_del[str(a)] = 0.
        flights_des_a = flights_hour[flights_hour['DEST_AIRPORT_ID'] == a]
        if len(flights_des_a) > 0:
            arr_del[str(a)] = flights_des_a['ARR_DELAY'].sum() / len(flights_des_a)
        else:
            arr_del[str(a)] = 0.
    return dep_del, arr_del

def turn_cols_to_int(df, cols):
    for c in cols:
        df[c] = df[c].astype(int)
    return df
