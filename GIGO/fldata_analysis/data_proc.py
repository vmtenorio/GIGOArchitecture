# Packages
import pandas as pd
import numpy as np
import os
from datetime import datetime
import data_utils
import graph_utils

# Consts
if os.name == 'nt':
    MAIN_PATH = '\\\\192.168.1.35\\Share\\Aero_TFG\\'
    slash = '\\'
else:
    MAIN_PATH = '/shared/Aero_TFG/'
    slash = '/'
DATA_PATH = MAIN_PATH + 'Datasets' + slash
OUT_PATH = MAIN_PATH + 'DataProc' + slash
DEBUG = True
log = open(OUT_PATH + 'log.txt', 'w')
#today = datetime.now().strftime('%Y%m{}-%H%M%')

loaded = False

# Read data

print('Reading the data...', end='', flush=True)
dfs = []
for dirname, dirs, files in os.walk(DATA_PATH):
    for d in dirs:
        filename = os.listdir(os.path.join(dirname, d))[0]
        full_path = os.path.join(dirname, d, filename)
        df_aux = pd.read_csv(full_path, dtype={'CRS_DEP_TIME': str})
        log.write("{}: {} flights read\n".format(d, str(len(df_aux.index))))
        dfs.append(df_aux)
df = pd.concat(dfs)
#df = pd.read_csv(DATA_PATH + jan_2019 + filename, dtype={'CRS_DEP_TIME': str})
#df = pd.read_csv('/home/meri/Escritorio/TFG Victor/TFG Aero/Data/517211445_T_ONTIME_REPORTING.csv')
print('DONE - Entries read: ' + str(len(df.index)), flush=True)

# Preprocessing

# Removing non-necesary columns
df = df[['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'FL_DATE', 'CRS_DEP_TIME', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DEP_DELAY', 'ARR_DELAY']]

print('Applying preprocessing')
# Convert dates to datetime objects
print('Converting dates to datetime objects')
df['FL_DATE'] = df['FL_DATE'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

#Removing airports with low flights
print('Reducing number of samples by removing airports with low number of flights')
df = data_utils.remove_airports(df, 10)
if DEBUG:
    print("Entries left: " + str(len(df.index)), flush=True)

# Get a list of unique airports after removing samples
print('Getting a list of airports. ', end='')
airports_uniq = pd.concat([df['ORIGIN_AIRPORT_ID'], df['DEST_AIRPORT_ID']]).drop_duplicates().reset_index(drop=True)
print("Unique Airports: " + str(len(airports_uniq)))
#print(str(airports_uniq))

# Convert hours to datetime objects
print('Converting hours to datetime objects')
df['CRS_DEP_TIME'] = df['CRS_DEP_TIME'].apply(lambda x: datetime.strptime(x, "%H%M"))
#print(df['CRS_DEP_TIME'])

if not loaded:
    # Create airport graph
    print('Creating airport graph')
    adj_mat = graph_utils.create_airport_graph(df, airports_uniq)
    #Normalize the matrix
    print('Normalizing')
    adj_mat = graph_utils.norm_graph(adj_mat)
    np.savetxt(OUT_PATH + 'adj_mat.csv', adj_mat, delimiter=',')


# Construct new dataframe with day preprocessing
cols_new = ['Year', 'Month', 'Day', 'Hour','Day_of_week']
# if loaded:
#     dep_delay_df = pd.read_csv(OUT_PATH + 'dep_delay.csv')
#     arr_delay_df = pd.read_csv(OUT_PATH + 'arr_delay.csv')
#     m_i = arr_delay_df['Month'].max()
#     df.drop(df[df['MONTH'] < m_i].index, inplace=True)
#     dep_delay_df.drop(dep_delay_df[dep_delay_df['Month'] == m_i].index, inplace=True)
#     arr_delay_df.drop(arr_delay_df[arr_delay_df['Month'] == m_i].index, inplace=True)
# else:
dep_delay_df = pd.DataFrame(columns=cols_new)
arr_delay_df = pd.DataFrame(columns=cols_new)

print('Getting the delay states to construct new dataframes')
#prim_delay_df = pd.DataFrame(columns=cols_new)
month_y = [12,2]
day_m = [31,28,31,30,31,30,31,31,30,31,30,31]
years = [2018,2019]
time_data = {}
# First day day of Jan 2018 was Tuesday
dow =  2            #TODO correct this
df2 = df.copy()
for i in range(len(years)):
    y = years[i]
    time_data['Year'] = y
    index_year = df2['YEAR'] == y
    df_year = df2[index_year]
    #print("Month {}: {} flights\n".format(1, len(df_year[df_year['MONTH'] == 1].index)))
    #print("Month {}: {} flights\n".format(2, len(df_year[df_year['MONTH'] == 2].index)))
    log.write("Year {}: {} flights\n".format(y, len(df_year.index)))
    m_f = month_y[i]
    for m in range(1,m_f+1):
        time_data['Month'] = m
        print('Month ' + str(m) + ': ', end="")
        days = day_m[m-1]
        index_month = df_year['MONTH'] == m
        df_month = df_year[index_month]
        log.write("Month {}: {} flights\n".format(m, len(df_month.index)))
        print("Year: {} - Month {}: {} flights".format(y, m, len(df_month.index)))
        #print("Index month ant: len = {}, sum = {}".format(len(index_month), index_month.sum()))
        for d in range(1,days+1):
            index_day = df_month['DAY_OF_MONTH'] == d
            df_day = df_month[index_day]
            log.write("Day {}: {} flights\n".format(d, len(df_day.index)))
            print(str(d) + ', ', end='', flush=True)
            time_data['Day'] = d
            time_data['Day_of_week'] = dow
            dow = (dow % 7) + 1
            print("Year: {} - Month {} - Day {}: {} flights".format(y, m, d, len(df_day.index)))
            for h in range(24):
                time_data['Hour'] = h
                flights = df_day[df_day['CRS_DEP_TIME'].apply(lambda x: x.hour) == h]
                dep_delay, arr_delay = data_utils.hour_proc(flights, airports_uniq)
                dep_delay.update(time_data)
                arr_delay.update(time_data)
                dep_delay_df = dep_delay_df.append(dep_delay, ignore_index=True)
                arr_delay_df = arr_delay_df.append(arr_delay, ignore_index=True)
                del dep_delay, arr_delay
            #print("Month {}: {} flights".format(m, len(df_month.index)))
            df_month = df_month[~index_day]
            #print("Month {}: {} flights".format(m, len(df_month.index)))
        #print("Index month desp: len = {}, sum = {}".format(len(index_month), index_month.sum()))
        #print(len(df_year.index))
        df_year = df_year[~index_month]
        #print(len(df_year.index))
        print("End month " + str(m))
    #df2.drop(df2[index_year].index, inplace=True)
    #df2.reset_index(drop=True, inplace=True)

print('END')

dep_delay_df = data_utils.turn_cols_to_int(dep_delay_df, cols_new)
arr_delay_df = data_utils.turn_cols_to_int(arr_delay_df, cols_new)

dep_delay_df.to_csv(OUT_PATH + 'dep_delay.csv', index=False, index_label=False)
arr_delay_df.to_csv(OUT_PATH + 'arr_delay.csv', index=False, index_label=False)

log.close()
