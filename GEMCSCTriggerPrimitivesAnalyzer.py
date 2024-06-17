import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_root_file(filename):
    """
    Reads a ROOT file and returns a dictionary of DataFrames for each TTree.

    Parameters:
        filename (str): Path to the ROOT file.

    Returns:
        dict: Dictionary with keys as TTree names and values as DataFrames.
    """
    file = uproot.open(filename)
    dataframes = {}
    for key in file.keys():
        if isinstance(file[key], uproot.behaviors.TTree.TTree):
            tree = file[key]
            df = tree.arrays(library="pd")
            dataframes[key] = df.reset_index()
    return dataframes

def match_LCTs(LCT_df, match_columns, loose_matching=None):
    """
    Matches data and emulator LCTs based on specified columns, with optional loose matching.

    Parameters:
        LCT_df (pd.DataFrame): DataFrame containing LCT data.
        match_columns (list): List of column names for exact matching.
        loose_matching (dict, optional): Dictionary specifying columns and tolerance values for loose matching.

    Returns:
        pd.DataFrame: DataFrame with matched LCTs, including a 'match' column.
    """
    # Separate data and emulator
    LCT_df = LCT_df.copy().reset_index()
    LCT_df_data = LCT_df[LCT_df['is_data']]
    LCT_df_emul = LCT_df[LCT_df['is_emul']]

    # Create a 'match' column and initialize with zeros
    LCT_df['match'] = 0
    match_count = 1

    # Exact matching columns
    exact_match_cols = [col for col in match_columns if col not in (loose_matching or {})]

    # Perform exact matching
    merged_df = pd.merge(LCT_df_data, LCT_df_emul, on=exact_match_cols, how='inner', suffixes=('_data', '_emul'))

    # Perform loose matching if specified
    if loose_matching:
        for col, tolerance in loose_matching.items():
            merged_df = merged_df[
                (merged_df[f"{col}_data"] >= (merged_df[f"{col}_emul"] - tolerance)) & 
                (merged_df[f"{col}_data"] <= (merged_df[f"{col}_emul"] + tolerance))
            ]

    # If there are matches, assign match numbers
    if not merged_df.empty:
        # Assign match numbers to the original DataFrame
        for index, row in merged_df.iterrows():
            match_num = match_count + index
            LCT_df.loc[LCT_df['index'] == row['index_data'], 'match'] = match_num
            LCT_df.loc[LCT_df['index'] == row['index_emul'], 'match'] = match_num
        
        # Increment match_count for the next batch of matches
        match_count += len(merged_df)

    return LCT_df

def filter_chamber(LCT_df, endcap='all', station='all', ring='all', chamber='all'):
    """
    Filters the LCT DataFrame based on specified chamber criteria.

    Parameters:
        LCT_df (pd.DataFrame): DataFrame containing LCT data.
        endcap (str): Endcap filter ('all' for no filter).
        station (str): Station filter ('all' for no filter).
        ring (str): Ring filter ('all' for no filter).
        chamber (str): Chamber filter ('all' for no filter).

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    LCT_df = LCT_df.copy()
    if endcap != 'all':
        LCT_df = LCT_df.query('endcap == @endcap')
    if station != 'all':
        LCT_df = LCT_df.query('station == @station')
    if ring != 'all':
        LCT_df = LCT_df.query('ring == @ring') 
    if chamber != 'all':
        LCT_df = LCT_df.query('chamber == @chamber')
    return LCT_df

def plot_LCT_parameter_histogram(LCT_df, parameter, endcap='all', station='all', ring='all', chamber='all'):
    """
    Plots a histogram of the specified LCT parameter for data and emulator.

    Parameters:
        LCT_df (pd.DataFrame): DataFrame containing LCT data.
        parameter (str): Parameter to plot.
        endcap (str): Endcap filter ('all' for no filter).
        station (str): Station filter ('all' for no filter).
        ring (str): Ring filter ('all' for no filter).
        chamber (str): Chamber filter ('all' for no filter).
    """
    LCT_df = filter_chamber(LCT_df, endcap, station, ring, chamber)
    LCT_df_data = LCT_df.query('is_data == 1')
    LCT_df_emul = LCT_df.query('is_emul == 1')

    plt.figure(figsize=(14, 3))
    bins = max(len(LCT_df_data[parameter].unique()), len(LCT_df_emul[parameter].unique()))
    range_vals = (min(LCT_df_data[parameter].min(), LCT_df_emul[parameter].min()), 
                  max(LCT_df_data[parameter].max(), LCT_df_emul[parameter].max()))

    plt.hist(LCT_df_data[parameter], bins=bins, range=range_vals, histtype='bar', alpha=0.7, label='data', align='mid', rwidth=0.8, color='red')
    plt.hist(LCT_df_emul[parameter], bins=bins, range=range_vals, histtype='bar', alpha=0.7, label='emul', align='mid', rwidth=0.6, color='blue')

    plt.xlabel(f'Parameter: {parameter}, Endcap: {endcap}, Station: {station}, Ring: {ring}, Chamber: {chamber}')
    plt.ylabel('Entries')
    plt.title('LCT Parameter Histogram')
    plt.legend()
    plt.show()

def plot_matched_LCT_count(LCT_df, endcap='all', station='all', ring='all', chamber='all'):
    """
    Plots a histogram of matched LCTs count for data and emulator.

    Parameters:
        LCT_df (pd.DataFrame): DataFrame containing LCT data.
        endcap (str): Endcap filter ('all' for no filter).
        station (str): Station filter ('all' for no filter).
        ring (str): Ring filter ('all' for no filter).
        chamber (str): Chamber filter ('all' for no filter).
    """
    LCT_df = filter_chamber(LCT_df, endcap, station, ring, chamber)
    LCT_df['match_count'] = LCT_df['match'].apply(lambda x: 1 if x > 0 else 0)
    LCT_df_data = LCT_df[LCT_df['is_data']]
    LCT_df_emul = LCT_df[LCT_df['is_emul']]

    plt.figure(figsize=(4, 3))
    plt.hist(LCT_df_data['match_count'], bins=2, range=(0, 1), histtype='bar', alpha=0.7, label='data', align='mid', rwidth=0.8)
    plt.hist(LCT_df_emul['match_count'], bins=2, range=(0, 1), histtype='bar', alpha=0.7, label='emul', align='mid', rwidth=0.6)

    plt.xticks([0.25, 0.75], ['No match', 'Match'])
    plt.xlabel(f'Matched LCTs count,\n Endcap: {endcap}, Station: {station}, Ring: {ring}, Chamber: {chamber}')
    plt.ylabel('Entries')
    plt.legend()
    plt.show()

def plot_heatmap_with_event_counts(LCT_df):
    """
    Plots a heatmap showing the ratio of 'data with match' to 'total data' for each chamber, with event counts annotated.

    Parameters:
        LCT_df (pd.DataFrame): DataFrame containing matched LCT data.
    """
    # Calculate the ratio of 'data with match' divided by 'total data' for each chamber
    ratio_data = LCT_df.groupby(['endcap', 'station', 'ring', 'chamber']).apply(
        lambda group: group[group['is_data']]['match'].apply(lambda x: 1 if x > 0 else 0).sum() / 
                      (group[group['is_data']]['match'].apply(lambda x: 1 if x == 0 else 0).sum() + 
                       group[group['is_data']]['match'].apply(lambda x: 1 if x > 0 else 0).sum())
    )

    # Calculate the number of events in each chamber from the data
    event_count = LCT_df[LCT_df['is_data']].groupby(['endcap', 'station', 'ring', 'chamber']).size()

    # Create a DataFrame with the calculated ratios and event counts
    ratio_df = pd.DataFrame({
        'Chamber': ratio_data.index.get_level_values('chamber'),
        'Endcap': ratio_data.index.get_level_values('endcap'),
        'Station': ratio_data.index.get_level_values('station'),
        'Ring': ratio_data.index.get_level_values('ring'),
        'Ratio': ratio_data.values,
        'EventCount': event_count.values
    })

    # Pivot the DataFrame for easy plotting
    ratio_pivot = ratio_df.pivot_table(index=['Chamber'], columns=['Endcap', 'Station', 'Ring'], values='Ratio')
    event_pivot = ratio_df.pivot_table(index=['Chamber'], columns=['Endcap', 'Station', 'Ring'], values='EventCount')

    # Convert the ratio_pivot DataFrame to a 2D NumPy array
    data_array = ratio_pivot.to_numpy()

    # Set up the Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create a heatmap using Matplotlib's pcolor
    cax = ax.pcolor(data_array, cmap='viridis', edgecolors='w', linewidths=0.5)

    # Add colorbar
    cbar = plt.colorbar(cax, label='Data with Match / All Data')

    # Set axis labels and title
    ax.set_title('2D Heatmap: Data with Match / All Data')
    ax.set_xlabel('Endcap - Station - Ring')
    ax.set_ylabel('Chamber')

    # Add chamber labels on the y-axis
    ax.set_yticks(np.arange(0.5, len(ratio_pivot.index), 1))
    ax.set_yticklabels(ratio_pivot.index)

    # Add endcap-station-ring labels on the x-axis
    ax.set_xticks(np.arange(0.5, len(ratio_pivot.columns), 1))
    ax.set_xticklabels(ratio_pivot.columns, rotation=90, ha='right')

    # Annotate the heatmap with event counts
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            if not np.isnan(data_array[i, j]):
                event_count = event_pivot.iloc[i, j]
                ax.text(j + 0.5, i + 0.5, f'{int(event_count)}', color='black', ha='center', va='center')

    plt.show()

