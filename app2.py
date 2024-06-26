from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from snowflake.connector import connect
import requests
from io import StringIO
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import base64
from io import BytesIO

app2 = Flask(__name__, template_folder = '.')
CORS(app2)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Connecting directly to Snowflake
conn = connect(
    user='BALAPP',
    password='HTflake1',
    account='uqa28410.us-east-1',
    database='YES',
    schema='YESDATA',
    warehouse='SMALL'
)

def read_sf(sql, conn=conn):
    cursor = conn.cursor()
    cursor.execute(sql)
    df = cursor.fetch_pandas_all()
    cursor.close()
    return df

def get_month_days(yearmo):
    year = int(yearmo[:4])
    month = yearmo[4:]
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    month_days = {
        '01': 31, '02': 29 if is_leap_year else 28, '03': 31, '04': 30,
        '05': 31, '06': 30, '07': 31, '08': 31, '09': 30, '10': 31,
        '11': 30, '12': 31
    }
    return month_days[month]

@app2.route('/')
def index():
    return render_template('index2.html')

@app2.route('/get_options', methods=['GET'])
def get_options():
    csv_url = 'https://raw.githubusercontent.com/bhuv1p/Capacity-Dashboard/main/companies_key.csv'
    access_token = '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'
    headers = {'Authorization': f'token {access_token}'}
    response = requests.get(csv_url, headers=headers)
    response.raise_for_status() 
    company_key = pd.read_csv(StringIO(response.text))

    company_key.fillna('Null', inplace=True)
    company_key.columns = company_key.columns.str.lower()
    company_key.columns = [col.replace(' ', '_') for col in company_key.columns]

    plant_ids = company_key['plant_id'].unique()
    plant_ids = [str(id) for id in plant_ids]
    merged_plant_ids = '(' + ','.join(f"'{plant}'" for plant in plant_ids) + ')'

    # Read data for CEMS from Snowflake
    date_for_code = "'" + f'{datetime.now().year - 5}' + '-01-01' + "'"
    part1_of_the_code = """select
    SUM(gen.value) AS gross_load,
    pl.plant_code AS plant_code,
    pl.plant_name AS plant_name,
    pl.state as state,
    TO_CHAR (datetime, 'YYYY-MM-DD HH') as date
    from TS_CEMS_PLANT_GEN_V gen
    left join DS_PLANTS pl
        ON pl.objectid = gen.objectid
    where
        TO_CHAR (datetime, 'YYYY-MM-DD') >="""
    part2_of_the_code = "and plant_code in"
    part3_of_the_code = """GROUP BY
    TO_CHAR (datetime, 'YYYY-MM-DD HH'),
    pl.plant_code,
    pl.plant_name,
    pl.state"""
    final_cems_code = (' ').join([part1_of_the_code, date_for_code, part2_of_the_code, merged_plant_ids, part3_of_the_code])

    #Read data from LPI Snowflake
    #splitting the LPI code into three parts
    part1_of_the_code = """select
    SUM(gen.value)/12 AS gross_load,
    pl.plant_code AS plant_code,
    pl.plant_name AS plant_name,
    pl.state as state,
    TO_CHAR (datetime, 'YYYY-MM-DD HH') as date
    from LPI_GEN gen
    left join object_relationships o1
        ON o1.objectid1 = gen.objectid
    left join DS_PLANTS pl
        ON o1.objectid2 = pl.objectid
    where
        TO_CHAR (datetime, 'YYYY-MM-DD') >="""
    part2_of_the_code = "and plant_code in"
    part3_of_the_code = """GROUP BY
    TO_CHAR (datetime, 'YYYY-MM-DD HH'),
    pl.plant_code,
    pl.plant_name,
    pl.state"""
    final_lpi_code = (' ').join([part1_of_the_code, date_for_code, part2_of_the_code, merged_plant_ids, part3_of_the_code])

    cems_5y = read_sf(final_cems_code)
    lpi_5y = read_sf(final_lpi_code)

    #converting the gross_load datatype to float
    lpi_5y['GROSS_LOAD'] = lpi_5y['GROSS_LOAD'].astype(float)
    cems_5y['GROSS_LOAD'] = cems_5y['GROSS_LOAD'].astype(float)

    #cleaning data for merging, bringing all column names to the same
    cems_5y.columns = cems_5y.columns.str.lower()
    lpi_5y.columns = lpi_5y.columns.str.lower()
    cems_5y.rename(columns={"plant_code":"plant_id", 'gross_load':'gross_load(mwh)'}, inplace=True)
    lpi_5y.rename(columns={"plant_code":"plant_id",'gross_load':'gross_load(mwh)'}, inplace=True)
    
    cems_5y['source'] = 'cems'
    lpi_5y['source'] = 'lpi'

    cems_5y[['year', 'month', 'day-hour']] = cems_5y['date'].str.split('-', expand=True)
    cems_5y[['day', 'hour']] = cems_5y['day-hour'].str.split(' ', expand=True)
    cems_5y['month'] = cems_5y['month'].str.zfill(2)
    cems_5y['day'] = cems_5y['day'].str.zfill(2)
    cems_5y['hour'] = cems_5y['hour'].str.zfill(2)
    cems_5y['yearmo'] = cems_5y['year'] + cems_5y['month']

    lpi_5y[['year', 'month', 'day-hour']] = lpi_5y['date'].str.split('-', expand=True)
    lpi_5y[['day', 'hour']] = lpi_5y['day-hour'].str.split(' ', expand=True)
    lpi_5y['month'] = lpi_5y['month'].str.zfill(2)
    lpi_5y['day'] = lpi_5y['day'].str.zfill(2)
    lpi_5y['hour'] = lpi_5y['hour'].str.zfill(2)
    lpi_5y['yearmo'] = lpi_5y['year'] + lpi_5y['month']

    # Merge with company_key to get capacity(mw)
    #merged_data = cems_5y.merge(company_key[['plant_id', 'capacity(mw)']], on="plant_id", how="left")

    #removing the yearmos with incomplete data (less than 30% of max data) in cems
    valid_yearmos = cems_5y.groupby(['yearmo'])['date'].count()[~(cems_5y.groupby(['yearmo'])['date'].count() < 0.3*(cems_5y.groupby(['yearmo'])['date'].count().max()))].index

    # dropping unnecessary columns
    cems_5y.drop(labels=['day-hour'], axis = 'columns', inplace = True)
    lpi_5y.drop(labels=['day-hour'], axis = 'columns', inplace = True)

    # Remove commas and convert capacity(mw) to float
    #merged_data['capacity(mw)'] = merged_data['capacity(mw)'].str.replace(',', '').astype(float)

    # filtering for only complete data months
    cems_5y = cems_5y[cems_5y['yearmo'].isin(valid_yearmos)]

    # filtering for only relevant lpi data months
    lpi_5y = lpi_5y[(lpi_5y['yearmo'] > cems_5y['yearmo'].max())]

    # clubbing both lpi and cems generation data
    gen_5y = pd.concat([cems_5y,lpi_5y], axis = 'rows', ignore_index = True)

    gen_5y.columns = [col.replace(' ', '_') for col in gen_5y.columns]

    """### Merging the data with company key"""

    # dropping plant name as it is available on snowflake
    company_key.drop(labels=['plant_name'], axis = 1, inplace = True)

    #merging both the generation data with company static data
    merged_data = gen_5y.merge(company_key, on="plant_id", how = "left")

    # creating a newcolum that has quarter
    merged_data['quarter'] = np.where(merged_data['month'].isin(['01', '02', '03']), 'Quarter 1',
                                      np.where(merged_data['month'].isin(['04', '05', '06']), 'Quarter 2',
                                               np.where(merged_data['month'].isin(['07', '08', '09']), 'Quarter 3',
                                                        'Quarter 4')))    
    """### Calculating capcity and coverage"""

    #rated generation: the maximum generation of a powerplant
    #merged_data['rated_generation'] = merged_data['yearmo'].apply(get_month_days)*merged_data['capacity(mw)']*24

    #number of days the plant was operational
    coverage_in_month = merged_data.groupby(['plant_id','yearmo'])['date'].count().reset_index()

    global current_year
    current_year = pd.Timestamp.now().year

    """### Creating To-Date and Full quarter logic"""

    todate_limit = str(pd.to_datetime(merged_data['date'].max()).month).zfill(2) + str(pd.to_datetime(merged_data['date'].max()).day).zfill(2)
    merged_data['month_day'] = merged_data['month'] + merged_data['day']
    #merged_data = merged_data[merged_data['year'] >= str(current_year - 1)]
    merged_data['to_date'] = merged_data['month_day'].apply(lambda x: x < todate_limit)
    merged_data_to_date = merged_data[merged_data['to_date'] == True]
    merged_data_to_date.drop(columns = ['to_date','month_day','date'], inplace=True)
    merged_data_2 = merged_data.copy()
    merged_data_2 = merged_data_2[(merged_data_2['year'] == str(current_year - 1)) & (merged_data_2['to_date'] == False)]
    merged_data_2['year'] = '2024'
    merged_data_2['source'] = 'lpi'
    merged_data_full_quarter = pd.concat([merged_data,merged_data_2], axis = 'rows')
    merged_data_full_quarter.drop(columns = ['to_date','month_day','date','yearmo'], inplace=True)
    merged_data_full_quarter['yearmo'] = merged_data_full_quarter['year'] + merged_data_full_quarter['month']
    merged_data.drop(columns = ['month_day','to_date'], inplace=True)

    global merged_data_month_plant_level
    global merged_data_month_plant_level_to_date

    merged_data_month_plant_level = merged_data_full_quarter.groupby(['yearmo','quarter','year','month','plant_id','plant_name','state','company','company_segment','iso','iso_sub_zone','type','capacity(mw)','source'])[['gross_load(mwh)']].sum().reset_index()
    merged_data_month_plant_level_to_date = merged_data_to_date.groupby(['yearmo','quarter','year','month','plant_id','plant_name','state','company','company_segment','iso','iso_sub_zone','type','capacity(mw)','source'])[['gross_load(mwh)']].sum().reset_index()

    merged_data_month_plant_level['capacity(mw)'] = merged_data_month_plant_level['capacity(mw)'].apply(lambda x: int(x.replace(',','')))
    merged_data_month_plant_level_to_date['capacity(mw)'] = merged_data_month_plant_level_to_date['capacity(mw)'].apply(lambda x: int(x.replace(',','')))

    #rated generation: the maximum generation of a powerplant
    merged_data_month_plant_level['rated_generation'] = (merged_data_month_plant_level['yearmo'].apply(get_month_days))*merged_data_month_plant_level['capacity(mw)']*24
    merged_data_month_plant_level_to_date['rated_generation'] = (merged_data_month_plant_level_to_date['yearmo'].apply(get_month_days))*merged_data_month_plant_level_to_date['capacity(mw)']*24

    merged_data_month_plant_level = merged_data_month_plant_level.merge(coverage_in_month, on=['plant_id','yearmo'], how='left')
    merged_data_month_plant_level_to_date = merged_data_month_plant_level_to_date.merge(coverage_in_month, on=['plant_id','yearmo'], how='left')

    merged_data_month_plant_level.rename(columns = {'date':'coverage'}, inplace=True)
    merged_data_month_plant_level_to_date.rename(columns = {'date':'coverage'}, inplace=True)

    """### Identifying the important ISO

    This section needs to be edited if need arises
    """

    merged_data_month_plant_level_to_date['iso'] = np.where(
        (merged_data_month_plant_level_to_date['iso'].isin(['ERCOT', 'PJM', 'ISO-NE'])),
        merged_data_month_plant_level_to_date['iso'],'NYISO,MISO,CAISO'
    )
    merged_data_month_plant_level_to_date['all_ba'] = 'All BAs'

    merged_data_month_plant_level['iso'] = np.where(
        (merged_data_month_plant_level['iso'].isin(['ERCOT', 'PJM', 'ISO-NE'])),
        merged_data_month_plant_level['iso'],'NYISO,MISO,CAISO'
    )
    merged_data_month_plant_level['all_ba'] = 'All BAs'

    #merged_data_month_plant_level['month_days'] = merged_data_month_plant_level['yearmo'].apply(get_month_days)

    current_year = int(merged_data_month_plant_level['year'].max())
    #global regions
    regions = (list(merged_data_month_plant_level['iso'].unique()) + list(merged_data_month_plant_level['all_ba'].unique()))
    regions.reverse()

    years = [str(current_year - x) for x in [0, 1, 2, 3, 4, 5]]

    """### Plotting ISO level"""

    #from ipywidgets import interact, fixed
    #global companies
    companies = merged_data_month_plant_level['company'].unique()
    companies = np.append(companies,'All')
    companies = list(companies)
    
    options = {
        'companies': companies,
        'isos': regions,
    }

    return jsonify(options)

@app2.route('/plot_capacity_factor', methods=['POST'])
def plot_capacity_factor():
    data = request.json
    company = data.get('company')
    iso = data.get('iso')

    def plot_capacity_factor_by_company(company, title):

        data = merged_data_month_plant_level.copy()
        current_yearmo = int(str(pd.Timestamp.now().year) + str(pd.Timestamp.now().month).zfill(2))
        data = data[data['yearmo'].astype(int)  <= current_yearmo]

        if 'All' != company:
            data = data[data['company'].isin(company)]

        gross_load_by_ba = data.groupby(['iso','yearmo','month','year', 'source'])['gross_load(mwh)'].sum().reset_index()
        rated_gen_by_ba = data.groupby(['iso','yearmo','month','year','source'])['rated_generation'].sum().reset_index()
        cf_data_by_ba = gross_load_by_ba.merge(rated_gen_by_ba, on = ['iso','yearmo','month','year','source'], how = 'left')
        cf_data_by_ba['cf'] = cf_data_by_ba['gross_load(mwh)']/cf_data_by_ba['rated_generation'] *100

        def plot_capacity_factor_for_one_iso(data, title, company):

            if title != "All BAs":
                plot_df = data[data['iso'].isin(title)]
            else:
                plot_df = data

            if plot_df.empty:
                print(f"No data available for {title}.")
                return

            plot_df['month'] = plot_df['month'].astype(int)
            fig, ax = plt.subplots(figsize=(8, 5))

            years_interest = [str(current_year - x) for x in [0, 1, 2]]
            years_average = [str(current_year - x) for x in [1, 2, 3, 4, 5]]

            for year in years_interest:
                temp_df = plot_df[plot_df['year'] == year].groupby(['year','month'])[['rated_generation','gross_load(mwh)']].sum().reset_index()
                temp_df['cf'] = (temp_df['gross_load(mwh)'] / temp_df['rated_generation']  * 100).round(2)
                sns.lineplot(data=temp_df, x='month', y='cf', label=year, linewidth=2.5, ax=ax)

            #for scatter
            temp_df_scatter = plot_df[plot_df['yearmo'] >= plot_df[plot_df['source'] == 'lpi']['yearmo'].min()]
            #temp_df_scatter.to_clipboard()
            temp_df_scatter = temp_df_scatter.groupby(['year','month'])[['rated_generation','gross_load(mwh)']].sum().reset_index()
            temp_df_scatter['cf'] = (temp_df_scatter['gross_load(mwh)'] / temp_df_scatter['rated_generation']  * 100).round(2)
            plt.scatter(data = temp_df_scatter, x = 'month', y = 'cf', label = 'Includes LPI data', marker = '*', s = 100)

            temp_df_average = plot_df[plot_df['year'].isin(years_average)]
            avg_data = temp_df_average.groupby(['month'])['cf'].mean().reset_index()
            sns.lineplot(data=avg_data, x='month', y='cf', label='Previous 5 Year Mean', linestyle='--', linewidth=2.5, ax=ax)

            # Calculate the minimum and maximum bounds for the shaded area
            lower_bound = temp_df_average.groupby(['month'])['cf'].min().reset_index()['cf']
            upper_bound = temp_df_average.groupby(['month'])['cf'].max().reset_index()['cf']
            months = temp_df_average['month'].sort_values().unique()

            # Fill between the minimum and maximum values
            ax.fill_between(months, lower_bound, upper_bound, color='lightblue', alpha=0.3)

            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([str(i) for i in range(1, 13)])
            ax.set_xlabel('Month', fontsize=10)
            ax.set_ylabel('Capacity Factor (%)', fontsize=10)
            ax.set_title(f'Company - {company},' + " " + f'BA - {title}' , fontsize=12)

            ax.grid(True)

            # Create a custom legend with the shaded area
            range_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='5-Year Min/Max Range')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(range_patch)  # add the custom patch to existing handles
            ax.legend(handles=handles, loc='upper right', fontsize=7)

            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()

            return image_base64

        image_base64 = plot_capacity_factor_for_one_iso(cf_data_by_ba, title, company)
        return image_base64
    image_base64 = plot_capacity_factor_by_company(company, iso)
    return jsonify({'image': image_base64})

@app2.route('/plot_by_type', methods=['POST'])
def plot_by_type():
    data = request.json
    company = data.get('company')
    iso = data.get('iso')

    def plot_plants_by_iso(company,iso):
      current_year #= int(year)
      region = iso
      data = merged_data_month_plant_level.copy()
      data = data[data['yearmo']  <= str(pd.Timestamp.now().year) + str(pd.Timestamp.now().month).zfill(2)]
    
      if 'All' != company:
        data = data[data['company'].isin(company)]
     
      if region != 'All BAs':
        item = 'iso'
        data = data[data['iso'].isin(iso)]
      else:
        item = 'all_ba'
    
      type2 = []
      type1 = 'Coal'
      type2.extend([s for s in data['type'].unique() if 'Coal' in s])
      data_coal = data[data['type'].isin(type2)]
              
      type2 = []
      type1 = 'Gas CCGT'
      type2.extend([s for s in data['type'].unique() if 'Gas CCGT' in s])
      data_ccgt = data[data['type'].isin(type2)]
              
      type2 = []
      type1 = 'Peaker'
      type2.extend([s for s in data['type'].unique() if 'CT' in s or 'ST' in s])
      data_ct = data[data['type'].isin(type2)]
    
      lpi_starts = data[data['source'] == 'lpi']['yearmo'].min()
    
      #splitting plant x month level data by coal and gas
      #merged_data_month_plant_level_coal = merged_data_month_plant_level[merged_data_month_plant_level['type'] == 'Coal']
      #merged_data_month_plant_level_ccgt = merged_data_month_plant_level[merged_data_month_plant_level['type'] == 'Gas CCGT']
      #merged_data_month_plant_level_ct = merged_data_month_plant_level[merged_data_month_plant_level['type'].isin(['Gas CT', 'Gas ST/CT'])]
    
      gross_load_by_ba_coal = data_coal.groupby([item,'yearmo','month','year'])['gross_load(mwh)'].sum().reset_index()
      gross_load_by_ba_ccgt = data_ccgt.groupby([item,'yearmo','month','year'])['gross_load(mwh)'].sum().reset_index()
      gross_load_by_ba_ct = data_ct.groupby([item,'yearmo','month','year'])['gross_load(mwh)'].sum().reset_index()
    
      rated_gen_by_ba_coal = data_coal.groupby([item,'yearmo','month','year'])['rated_generation'].sum().reset_index()
      rated_gen_by_ba_ccgt = data_ccgt.groupby([item,'yearmo','month','year'])['rated_generation'].sum().reset_index()
      rated_gen_by_ba_ct = data_ct.groupby([item,'yearmo','month','year'])['rated_generation'].sum().reset_index()
    
      cf_data_by_ba_coal = gross_load_by_ba_coal.merge(rated_gen_by_ba_coal, on = [item,'yearmo','month','year'], how = 'left')
      cf_data_by_ba_ccgt = gross_load_by_ba_ccgt.merge(rated_gen_by_ba_ccgt, on = [item,'yearmo','month','year'], how = 'left')
      cf_data_by_ba_ct = gross_load_by_ba_ct.merge(rated_gen_by_ba_ct, on = [item,'yearmo','month','year'], how = 'left')
    
      #calculating cf by coal and gas
      cf_data_by_ba_coal['cf'] = cf_data_by_ba_coal['gross_load(mwh)']/cf_data_by_ba_coal['rated_generation'] * 100
      cf_data_by_ba_ccgt['cf'] = cf_data_by_ba_ccgt['gross_load(mwh)']/cf_data_by_ba_ccgt['rated_generation'] * 100
      cf_data_by_ba_ct['cf'] = cf_data_by_ba_ct['gross_load(mwh)']/cf_data_by_ba_ct['rated_generation'] * 100
    
      plot_df_coal = cf_data_by_ba_coal
      plot_df_ccgt = cf_data_by_ba_ccgt
      plot_df_ct = cf_data_by_ba_ct

      
      # Function to plot the data for a given DataFrame
      def plot_capacity_factor(plot_df, title, ax, current_year, item, region, lpi_starts):
          # Define the years of interest and average
    
          years_interest = [str(current_year - x) for x in [0, 1, 2]]
          years_average = [str(current_year - x) for x in [1, 2, 3, 4, 5]]
    
          # Plot the data for each year of interest
          for year in years_interest:
              sns.lineplot(data=plot_df[plot_df['year'] == year], x='month', y='cf', label=year, linewidth=2.5, ax=ax)
    
          # plotting lpi data
          temp_data = plot_df[plot_df['yearmo'] >= lpi_starts]
          temp_df_scatter = temp_data.groupby(['year','month'])[['rated_generation','gross_load(mwh)']].sum().reset_index()
          temp_df_scatter['cf'] = (temp_df_scatter['gross_load(mwh)'] / temp_df_scatter['rated_generation']  * 100).round(2)
          sns.scatterplot(data = temp_df_scatter, x = 'month', y = 'cf', label = 'Includes LPI data', marker = '*', s= 300, ax=ax)
    
          # Plot the average data for the previous 5 years
          plot_df_average = plot_df[plot_df['year'].isin(years_average)]
          avg_data = plot_df_average.groupby([item,'month'])['cf'].mean().reset_index()
          sns.lineplot(data=avg_data, x='month', y='cf', label='Previous 5 Year Mean', linestyle='--', linewidth=2.5, ax=ax)
    
          # Calculate the upper and lower bounds for the shaded area
          lower_bound = plot_df_average.groupby([item,'month'])['cf'].min().reset_index()['cf']
          upper_bound = plot_df_average.groupby([item,'month'])['cf'].max().reset_index()['cf']
          months = plot_df_average.groupby([item,'month'])['cf'].max().reset_index()['month']
    
          if not months.empty and not lower_bound.isnull().any() and not upper_bound.isnull().any():
            ax.fill_between(months, lower_bound, upper_bound, color='lightblue', alpha=0.3)
    
          # Rotate the x-axis labels
          ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
          # Add labels and title
          ax.set_xlabel('Month', fontsize=12)
          ax.set_ylabel('Capacity Factor (%)', fontsize=12)
          ax.set_title(f'{company} : {region} : {title}', fontsize=15)
    
          # Create a custom legend for the shaded area
          range_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='Previous 5 Year Range')
    
          # Get the current legend handles and labels
          handles, labels = ax.get_legend_handles_labels()
    
          # Append the custom legend patch for the shaded area
          handles.append(range_patch)
          labels.append('Previous 5 Year Range')
    
          # Add the combined legend to the plot
          ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=10)
          ax.grid(True)

    # Create a figure with 3 subplots side by side
      fig, axs = plt.subplots(1, 3, figsize=(18, 6))

      # Plot the data for each DataFrame
      plot_capacity_factor(plot_df_coal, 'Coal Plants', axs[0], current_year,item,region,lpi_starts)
      plot_capacity_factor(plot_df_ccgt, 'CCGT Plants', axs[1], current_year,item,region,lpi_starts)
      plot_capacity_factor(plot_df_ct, 'CT Plants', axs[2], current_year,item,region,lpi_starts)

      buf = BytesIO()
      plt.savefig(buf, format='png')
      buf.seek(0)
      image_base64 = base64.b64encode(buf.read()).decode('utf-8')
      buf.close()
      return image_base64
    image_base64 = plot_plants_by_iso(company, iso)
    return jsonify({'image': image_base64})

if __name__ == '__main__':
    app2.run(debug=True)
