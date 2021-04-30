# flowVarsDaily.py - Grabs daily data from USGS and calculates previous day flow variables
# Source: http://waterdata.usgs.gov/ca/nwis/current/?type=flow
# RTS - March 2018 UPDATE 3/23/21

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os

### Inputs
sd = '2010-01-01'  # YYYY-MM-DD format, include one day previous
ed = '2020-12-31'
outfolder = '../data/flow'

stations = {
            # 'Carmel River': '11143250',  # Not near Carmel City Beach
            # 'Salinas River': '11152500',  # Monterey/Marina
            'San Lorenzo River': '11161000',  # Cowell, Santa Cruz beaches
            'Soquel Creek': '11160000',  # Cowell, Santa Cruz beaches
            'Pescadero Creek': '11162500',  # Pescadero
            # 'Pilarcitos Creek': '11162630',  # Half moon bay
            # 'Redwood Creek': '11460151',  # Muir Beach (Marin)
            # 'Little River': '11481200'  # HB (Moonstone)
}

print('Flow Data\nDirectory: ' + outfolder)
df_summary = pd.DataFrame()
plt.figure(figsize=(10,5))
for key in stations:
    station_name = key
    station_no = stations[key]
    print('\nGrabbing flow data for ' + station_name)

    ### Grab USGS daily discharge data over the specified timeframe
    url = 'http://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=' + station_no + \
          '&referred_module=sw&period=&begin_date=' + sd + '&end_date=' + ed

    web = requests.get(url)
    try:
        web.raise_for_status()
    except Exception as exc:
        print('  There was a problem grabbing flow data: %s' % exc)

    data = [line.split() for line in web.text.splitlines()]
    while data[0][0].startswith('#'):  # delete comments from list
        del data[0]

    df = pd.DataFrame(data, columns=data[0]).drop([0, 1])  # Delete headers
    df = df[list(df.columns[2:])]  # Grab datetime, flow, and qualifier
    df.columns = ['date', 'flow', 'qual']
    df['date'] = pd.to_datetime(df['date'])
    df['flow'] = pd.to_numeric(df.flow, errors='coerce')
    df['logflow'] = np.log10(df.flow)
    df['quartile'] = 4            # Indicate which quantile of the record the flow data are in
    df.loc[df.flow < df.flow.quantile(.75),'quartile'] = 3
    df.loc[df.flow < df.flow.quantile(.5),'quartile'] = 2
    df.loc[df.flow < df.flow.quantile(.25),'quartile'] = 1
    df.set_index('date', inplace=True)
    print(' Data found')


    ### Save data
    outfile = station_name.replace(' ', '_').lower() + '_usgs_flow_' + \
    sd.replace('-', '') + '_' + ed.replace('-', '') + '.csv'
    
    df = df[['flow', 'logflow', 'quartile', 'qual']]
    df.to_csv(os.path.join(outfolder, outfile))
    print(' Raw data saved to ' + outfile)

    ### Plot time series
    plt.plot(df['logflow'], label=key)

    ### Summary of data
    missing = len(pd.date_range(sd,ed)) - len(df)

    sum_dict = {
        'ID': station_no,
        'Start Date': str(df.index[0].date()),
        'End Date': str(df.index[-1].date()),
        'Missing Days': missing
    }
    df_summary = df_summary.append(pd.DataFrame(sum_dict, index=[station_name]))

df_summary = df_summary[['ID', 'Start Date', 'End Date', 'Missing Days']]
df_summary.index.rename('Station', inplace=True)
print(df_summary)

### Plot 
plt.ylabel('Daily Mean Discharge [$log_{10}(ft^3/s)$]')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outfolder,'logflow_sites.png'),dpi=300)
