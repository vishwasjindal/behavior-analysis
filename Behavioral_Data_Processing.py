#importing all libraries
import os
import streamlit as st
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import git

#st.set_page_config(layout="wide")

#listing subfolders
def list_subfolders(path):
    subfolders = [f.name for f in os.scandir(path) if f.is_dir()]
    return subfolders
#listing the files
def list_files(path, prefix):
    files = [f.name for f in os.scandir(path) if f.is_file() and f.name.startswith(prefix)]
    return sorted(files, key=lambda f: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', f)])

#reading each file
def process_file(file_path):
    prefix = "P1" if "P1" in file_path else "P2"
    subdirectory_path = os.path.dirname(file_path)
    file_names = [os.path.basename(file_path)]
    phase_type = prefix
    files = []
    store_results = {}
    data = {}
    for f in file_names:
        files.append(os.path.join(subdirectory_path,f))
    for fileaddress in files:
        with open(fileaddress, 'r') as f:
            if phase_type == 'P1':
                data = np.loadtxt(f, dtype={'names': ('date', 'timestamp', 'time', 'averageforce', 'filteredforce', 'waterpump', 'ITI', 'threshold'), 
                                          'formats': ('S10', 'S10', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')})
                data = data[2:]
                fileaddress = fileaddress[3:-4]
                store_results[fileaddress] = data
            else:
                data = np.loadtxt(f, dtype={'names': ('date', 'timestamp', 'time', 'averageforce', 'filteredforce', 'waterpump', 'ITI', 'threshold'), 
                                          'formats': ('S10', 'S10', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')})
                data = data[2:]
                fileaddress = fileaddress[3:-4]
                store_results[fileaddress] = data
    return store_results

########################### Phase 1 functions ###################################################
#processing data
def analyze_data_P1(file_path):
    # your existing code
    animal_name = file_path
    min_time_bet_2_press = 20 # of samples; 20 x 10ms = 200 ms
    total_presses = {}
    total_rewards = {}
    success_force_rewarded = {}
    mean_force_rewarded = {}
    std_force_rewarded = {}
    average_time_between_each_press = {}
    average_time_between_rewarded_press = {}
    mean_press_time_to_ITI = {}
    std_press_time_to_ITI = {}
    total_presses[animal_name] = []
    total_rewards[animal_name] = []
    mean_force_rewarded[animal_name] = []
    std_force_rewarded[animal_name] = []
    average_time_between_each_press[animal_name] = []
    average_time_between_rewarded_press[animal_name] = []
    mean_press_time_to_ITI[animal_name] = []
    std_press_time_to_ITI[animal_name] = []
    success_force_rewarded[animal_name] = []
    store_results = process_file(file_path)
    single_file_name = list(store_results.keys())[0]
    data = store_results[single_file_name]
    loc_high_force = np.where(data['filteredforce'] > data['threshold'])[0]
    diff_loc = np.diff(loc_high_force)
    # identify the locations and times of press onsets
    press_onset_loc = loc_high_force[np.concatenate(([0], np.where(diff_loc >= min_time_bet_2_press)[0] + 1))]
    press_onset_time = data['time'][press_onset_loc]
    tot_num_press = len(press_onset_loc) #Param1
    total_presses[animal_name].append(tot_num_press)
    ##########################################################################################
    #Rewards
    water_on_loc = np.where(data['waterpump'] == 1)[0]
    diff_water_loc = np.diff(water_on_loc)
    # Select locations of water rewards
    water_reward_loc = water_on_loc[np.concatenate(([0], np.where(diff_water_loc > 1)[0]+1))]
    # Calculate total number of water rewards
    tot_num_water_reward = len(water_reward_loc) #param 2
    total_rewards[animal_name].append(tot_num_water_reward)
        # Determine number of rewards based of presses
    force_at_water = data['filteredforce'][water_reward_loc-1]
    threshold_at_water =  data['threshold'][water_reward_loc-1]
    success_loc = water_reward_loc[np.where(force_at_water > threshold_at_water)[0]]
    success_press = np.sum(force_at_water > threshold_at_water) #param3
    success_force_rewarded[animal_name].append(success_press)
    #######################################################################################################
    #force during thr rewards
    max_press_period = 15 #20 * 10ms = 200ms
    force_rewarded = np.zeros(success_press)
    for j in range(success_press):            
        force_rewarded[j] = np.max(data['filteredforce'][success_loc[j] - 1: success_loc[j] + max_press_period])
    mean_force_reward = np.mean(force_rewarded) #param4
    std_force_reward = np.std(force_rewarded) #param5
    mean_force_rewarded[animal_name].append(mean_force_reward)
    std_force_rewarded[animal_name].append(std_force_reward)
    ##########################################################################################################
    #Time between presses
    time_between_each_press = np.diff(press_onset_time)
    avg_time_between_each_press = np.mean(time_between_each_press) # unit:sec Param 6
    average_time_between_each_press[animal_name].append(avg_time_between_each_press)
    #average gtime between rewarded press
    time_rewarded_press = data['time'][success_loc]
    time_between_rewarded_press = np.insert(np.diff(time_rewarded_press), 0, time_rewarded_press[0])
    avg_time_between_rewarded_press = np.mean(time_between_rewarded_press) # Param 7
    average_time_between_rewarded_press[animal_name].append(avg_time_between_rewarded_press)
    ####################################################################################################
    #ITI
    ITI_rewarded_press = data['ITI'][success_loc] / 1000
    press_time_to_ITI_ratio = time_between_rewarded_press / ITI_rewarded_press
    #REMOVING OUTLIERS
    outliers_ratios = np.where((press_time_to_ITI_ratio > np.mean(press_time_to_ITI_ratio) + 3*np.std(press_time_to_ITI_ratio)) | (press_time_to_ITI_ratio < np.mean(press_time_to_ITI_ratio) - 3*np.std(press_time_to_ITI_ratio)))[0]
    press_time_to_ITI_ratio_no_outlier = np.delete(press_time_to_ITI_ratio, outliers_ratios)
    mean_press_time_to_ITI_ratio_no_outlier = np.mean(press_time_to_ITI_ratio_no_outlier) #PARAM 8
    mean_press_time_to_ITI[animal_name].append(mean_press_time_to_ITI_ratio_no_outlier.tolist())
    std_press_time_to_ITI_ratio_no_outlier = np.std(press_time_to_ITI_ratio_no_outlier) #PARAM 9
    std_press_time_to_ITI[animal_name].append(std_press_time_to_ITI_ratio_no_outlier.tolist())
    result = {}
    animal_name = file_path
    result['total_presses'] = total_presses[animal_name]
    result['total_rewards'] = total_rewards[animal_name]
    result['success_force_rewarded'] = success_force_rewarded[animal_name]
    result['mean_force_rewarded'] = mean_force_rewarded[animal_name]
    result['std_force_rewarded'] = std_force_rewarded[animal_name]
    result['average_time_between_each_press'] = average_time_between_each_press[animal_name]
    result['average_time_between_rewarded_press'] = average_time_between_rewarded_press[animal_name]
    result['mean_press_time_to_ITI'] = mean_press_time_to_ITI[animal_name]
    result['std_press_time_to_ITI'] = std_press_time_to_ITI[animal_name]
    return result

# extracting all results for Phase 1
def extract_metrics_P1(processed_data, animal_id):
    metrics = {
        'total_presses': [],
        'total_rewards': [],
        'success_force_rewarded': [],
        'mean_force_rewarded': [],
        'std_force_rewarded': [],
        'average_time_between_each_press': [],
        'average_time_between_rewarded_press': [],
        'mean_press_time_to_ITI': [],
        'std_press_time_to_ITI': []
    }
    
    for day in range(0, 40):
        key = f"{animal_id}_Day{day}"
        try:
            data = processed_data[key]
        except KeyError:
            print(f"Warning: {key} not found in processed_data")
            continue
        
        metrics['total_presses'].append(data['total_presses'][0])
        metrics['total_rewards'].append(data['total_rewards'][0])
        metrics['success_force_rewarded'].append(data['success_force_rewarded'][0])
        metrics['mean_force_rewarded'].append(data['mean_force_rewarded'][0])
        metrics['std_force_rewarded'].append(data['std_force_rewarded'][0])
        metrics['average_time_between_each_press'].append(data['average_time_between_each_press'][0])
        metrics['average_time_between_rewarded_press'].append(data['average_time_between_rewarded_press'][0])
        metrics['mean_press_time_to_ITI'].append(data['mean_press_time_to_ITI'][0])
        metrics['std_press_time_to_ITI'].append(data['std_press_time_to_ITI'][0])
    
    return metrics


#plotting data for every animal individually Phase1
def plot_metrics_P1(processed_data, animal_id, animal_name):
    # Extract the metrics
    animal_data = extract_metrics_P1(processed_data, animal_id)
    
    # Define dictionary keys and corresponding plot titles
    keys = ['total_presses', 'total_rewards', 'success_force_rewarded', 'mean_force_rewarded', 'std_force_rewarded', 'average_time_between_each_press', 'average_time_between_rewarded_press', 'mean_press_time_to_ITI', 'std_press_time_to_ITI']
    titles = ['Total Presses', 'Total Rewards', 'Successful Presses with Rewards', 'Mean Force of Successful Presses', 'Standard Deviation of Force of Successful Presses', 'Average Time Between Each Press', 'Average Time Between Rewarded Press', 'Mean Press Time to ITI', 'Standard Deviation of Press Time to ITI']
    
    # Loop over the metrics and plot each one using Streamlit
    for i, key in enumerate(keys):
        st.write(f"{titles[i]} for {animal_name}")
        chart_data = animal_data[key]
        days = range(1, len(chart_data) + 1)
        plt.plot(days, chart_data)
        plt.xlabel("Day")
        plt.ylabel(f"{titles[i]}")
        st.pyplot(plt.gcf())
        plt.clf()
        

#Plotting entire data
def plot_all_metrics_P1(processed_data, animal_ids, animal_names):
    # Define dictionary keys and corresponding plot titles
    keys = ['total_presses', 'total_rewards', 'success_force_rewarded', 'mean_force_rewarded', 'std_force_rewarded', 'average_time_between_each_press', 'average_time_between_rewarded_press', 'mean_press_time_to_ITI', 'std_press_time_to_ITI']
    titles = ['Total Presses', 'Total Rewards', 'Successful Presses with Rewards', 'Mean Force of Successful Presses', 'Standard Deviation of Force of Successful Presses', 'Average Time Between Each Press', 'Average Time Between Rewarded Press', 'Mean Press Time to ITI', 'Standard Deviation of Press Time to ITI']

    # Loop over the metrics and plot each one using Streamlit
    for i, key in enumerate(keys):
        plt.figure()
        plt.title(f"{titles[i]} for all animals")
        plt.xlabel("Day")
        plt.ylabel(f"{titles[i]}")

        for j, animal_id in enumerate(animal_ids):
            animal_data = extract_metrics_P1(processed_data, animal_id)
            chart_data = animal_data[key]
            days = range(1, len(chart_data) + 1)
            plt.plot(days, chart_data, label=animal_names[j])

        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()
#################################################################################################
############### Phase 2 functions ##################
#processing data
def analyze_data_P2(file_path):
    # your existing code
    animal_name = file_path
    min_time_bet_2_press = 20 # of samples; 20 x 10ms = 200 ms
    total_presses = {}
    total_rewards = {}
    success_force_rewarded = {}
    mean_force_rewarded = {}
    std_force_rewarded = {}
    average_time_between_each_press = {}
    average_time_between_rewarded_press = {}
    mean_press_time_to_ITI = {}
    std_press_time_to_ITI = {}
    total_presses[animal_name] = []
    total_rewards[animal_name] = []
    mean_force_rewarded[animal_name] = []
    std_force_rewarded[animal_name] = []
    average_time_between_each_press[animal_name] = []
    average_time_between_rewarded_press[animal_name] = []
    mean_press_time_to_ITI[animal_name] = []
    std_press_time_to_ITI[animal_name] = []
    success_force_rewarded[animal_name] = []
    store_results = process_file(file_path)
    single_file_name = list(store_results.keys())[0]
    data = store_results[single_file_name]
    loc_high_force = np.where(data['filteredforce'] > data['threshold'])[0]
    diff_loc = np.diff(loc_high_force)
    # identify the locations and times of press onsets
    press_onset_loc = loc_high_force[np.concatenate(([0], np.where(diff_loc >= min_time_bet_2_press)[0] + 1))]
    press_onset_time = data['time'][press_onset_loc]
    tot_num_press = len(press_onset_loc) #Param1
    total_presses[animal_name].append(tot_num_press)
    ##########################################################################################
    #Rewards
    water_on_loc = np.where(data['waterpump'] == 1)[0]
    diff_water_loc = np.diff(water_on_loc)
    # Select locations of water rewards
    water_reward_loc = water_on_loc[np.concatenate(([0], np.where(diff_water_loc > 1)[0]+1))]
    # Calculate total number of water rewards
    tot_num_water_reward = len(water_reward_loc) #param 2
    total_rewards[animal_name].append(tot_num_water_reward)
        # Determine number of rewards based of presses
    force_at_water = data['filteredforce'][water_reward_loc-1]
    threshold_at_water =  data['threshold'][water_reward_loc-1]
    success_loc = water_reward_loc[np.where(force_at_water > threshold_at_water)[0]]
    success_press = np.sum(force_at_water > threshold_at_water) #param3
    success_force_rewarded[animal_name].append(success_press)
    #######################################################################################################
    #force during thr rewards
    max_press_period = 15 #20 * 10ms = 200ms
    force_rewarded = np.zeros(success_press)
    for j in range(success_press):            
        force_rewarded[j] = np.max(data['filteredforce'][success_loc[j] - 1: success_loc[j] + max_press_period])
    mean_force_reward = np.mean(force_rewarded) #param4
    std_force_reward = np.std(force_rewarded) #param5
    mean_force_rewarded[animal_name].append(mean_force_reward)
    std_force_rewarded[animal_name].append(std_force_reward)
    ##########################################################################################################
    #Time between presses
    time_between_each_press = np.diff(press_onset_time)
    avg_time_between_each_press = np.mean(time_between_each_press) # unit:sec Param 6
    average_time_between_each_press[animal_name].append(avg_time_between_each_press)
    #average gtime between rewarded press
    time_rewarded_press = data['time'][success_loc]
    time_between_rewarded_press = np.insert(np.diff(time_rewarded_press), 0, time_rewarded_press[0])
    avg_time_between_rewarded_press = np.mean(time_between_rewarded_press) # Param 7
    average_time_between_rewarded_press[animal_name].append(avg_time_between_rewarded_press)
    ####################################################################################################
    #ITI
    ITI_rewarded_press = data['ITI'][success_loc] / 1000
    press_time_to_ITI_ratio = time_between_rewarded_press / ITI_rewarded_press
    #REMOVING OUTLIERS
    outliers_ratios = np.where((press_time_to_ITI_ratio > np.mean(press_time_to_ITI_ratio) + 3*np.std(press_time_to_ITI_ratio)) | (press_time_to_ITI_ratio < np.mean(press_time_to_ITI_ratio) - 3*np.std(press_time_to_ITI_ratio)))[0]
    press_time_to_ITI_ratio_no_outlier = np.delete(press_time_to_ITI_ratio, outliers_ratios)
    mean_press_time_to_ITI_ratio_no_outlier = np.mean(press_time_to_ITI_ratio_no_outlier) #PARAM 8
    mean_press_time_to_ITI[animal_name].append(mean_press_time_to_ITI_ratio_no_outlier.tolist())
    std_press_time_to_ITI_ratio_no_outlier = np.std(press_time_to_ITI_ratio_no_outlier) #PARAM 9
    std_press_time_to_ITI[animal_name].append(std_press_time_to_ITI_ratio_no_outlier.tolist())
    result = {}
    animal_name = file_path
    result['total_presses'] = total_presses[animal_name]
    result['total_rewards'] = total_rewards[animal_name]
    result['success_force_rewarded'] = success_force_rewarded[animal_name]
    result['mean_force_rewarded'] = mean_force_rewarded[animal_name]
    result['std_force_rewarded'] = std_force_rewarded[animal_name]
    result['average_time_between_each_press'] = average_time_between_each_press[animal_name]
    result['average_time_between_rewarded_press'] = average_time_between_rewarded_press[animal_name]
    result['mean_press_time_to_ITI'] = mean_press_time_to_ITI[animal_name]
    result['std_press_time_to_ITI'] = std_press_time_to_ITI[animal_name]
    return result

# extracting all metrices for Phase 2
def extract_metrics_P2(processed_data, animal_id):
    metrics = {
        'total_presses': [],
        'total_rewards': [],
        'success_force_rewarded': [],
        'mean_force_rewarded': [],
        'std_force_rewarded': [],
        'average_time_between_each_press': [],
        'average_time_between_rewarded_press': [],
        'mean_press_time_to_ITI': [],
        'std_press_time_to_ITI': []
    }
    
    for day in range(0, 40):
        key = f"{animal_id}_Day{day}"
        try:
            data = processed_data[key]
        except KeyError:
            print(f"Warning: {key} not found in processed_data")
            continue
        # process data for the current day

    # process data for the current day

        
        metrics['total_presses'].append(data['total_presses'][0])
        metrics['total_rewards'].append(data['total_rewards'][0])
        metrics['success_force_rewarded'].append(data['success_force_rewarded'][0])
        metrics['mean_force_rewarded'].append(data['mean_force_rewarded'][0])
        metrics['std_force_rewarded'].append(data['std_force_rewarded'][0])
        metrics['average_time_between_each_press'].append(data['average_time_between_each_press'][0])
        metrics['average_time_between_rewarded_press'].append(data['average_time_between_rewarded_press'][0])
        metrics['mean_press_time_to_ITI'].append(data['mean_press_time_to_ITI'][0])
        metrics['std_press_time_to_ITI'].append(data['std_press_time_to_ITI'][0])
    
    return metrics


#plotting data for every animal individually Phase1
def plot_metrics_P2(processed_data, animal_id, animal_name):
    # Extract the metrics
    animal_data = extract_metrics_P2(processed_data, animal_id)
    #st.write(animal_data)
    
    # Define dictionary keys and corresponding plot titles
    keys = ['total_presses', 'total_rewards', 'success_force_rewarded', 'mean_force_rewarded', 'std_force_rewarded', 'average_time_between_each_press', 'average_time_between_rewarded_press', 'mean_press_time_to_ITI', 'std_press_time_to_ITI']
    titles = ['Total Presses', 'Total Rewards', 'Successful Presses with Rewards', 'Mean Force of Successful Presses', 'Standard Deviation of Force of Successful Presses', 'Average Time Between Each Press', 'Average Time Between Rewarded Press', 'Mean Press Time to ITI', 'Standard Deviation of Press Time to ITI']
    
    # Loop over the metrics and plot each one using Streamlit
    for i, key in enumerate(keys):
        st.write(f"{titles[i]} for {animal_name}")
        chart_data = animal_data[key]
        #st.write(chart_data)
        days = range(1, len(chart_data) + 1)
        plt.plot(days, chart_data)
        plt.xlabel("Day")
        plt.ylabel(f"{titles[i]}")
        st.pyplot(plt.gcf())
        plt.clf()
        
#Plotting entire data
       
def plot_all_metrics_P2(processed_data, animal_ids, animal_names):
    # Define dictionary keys and corresponding plot titles
    keys = ['total_presses', 'total_rewards', 'success_force_rewarded', 'mean_force_rewarded', 'std_force_rewarded', 'average_time_between_each_press', 'average_time_between_rewarded_press', 'mean_press_time_to_ITI', 'std_press_time_to_ITI']
    titles = ['Total Presses', 'Total Rewards', 'Successful Presses with Rewards', 'Mean Force of Successful Presses', 'Standard Deviation of Force of Successful Presses', 'Average Time Between Each Press', 'Average Time Between Rewarded Press', 'Mean Press Time to ITI', 'Standard Deviation of Press Time to ITI']

    # Loop over the metrics and plot each one using Streamlit
    for i, key in enumerate(keys):
        plt.figure()
        plt.title(f"{titles[i]} for all animals")
        plt.xlabel("Day")
        plt.ylabel(f"{titles[i]}")

        for j, animal_id in enumerate(animal_ids):
            animal_data = extract_metrics_P2(processed_data, animal_id)
            chart_data = animal_data[key]
            days = range(1, len(chart_data) + 1)
            plt.plot(days, chart_data, label=animal_names[j])

        plt.legend()
        st.pyplot(plt.gcf())
        plt.clf()


############################## Main function ####################################################      
def main():
    st.title("Behavioral Data Analysis")
    import requests
    
    address = "https://github.com/vishwasjindal/behavior-analysis/tree/main/Mousedata?access_token=<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>"


    # Check if the address exists
    response = requests.get(address)
    if response.status_code == 404:
        st.error("Invalid address")
        return


    st.write(f"Using GitHub folder: {address}")
    #Check if the address exists
    if address:
        if not os.path.exists(address):
            st.error("Invalid address")
            return
        subfolders = list_subfolders(address)
        #creating radiobutton of each subfolder
        selected_subfolder = st.radio("Select subfolder:", subfolders)
        subfolder_path = os.path.join(address, selected_subfolder)
        #showing all files reated to Phase 1 and 2
        phase_prefix = st.radio("Select phase prefix:", ["P1", "P2"])
        phase_files = list_files(subfolder_path, phase_prefix)
        if len(phase_files) == 0:
            st.warning("No files found")
            return
        #SELECTING Phase1 or Phase2
        selected_file = st.radio("Select file:", phase_files)
        st.write(f"{selected_file}")
        file_path = os.path.join(subfolder_path, selected_file)
        st.write(f"{file_path}")
        #When phase1 clicked
        if phase_prefix == 'P1':
            processed_data = analyze_data_P1(file_path)
            df = pd.DataFrame(processed_data)
            #starting index with 1
            df.index = pd.RangeIndex(start=1, stop=len(df)+1)
            # Display the table
            st.table(df)
        #HEN PHASE2 CLICKED    
        elif phase_prefix == 'P2':
            processed_data = analyze_data_P2(file_path)
            df = pd.DataFrame(processed_data)
            #starting index with 1
            df.index = pd.RangeIndex(start=1, stop=len(df)+1)
            # display the dataframe as a table in Streamlit
            st.table(df)
        # Button to load all files in subfolder PHASE1
        if st.button("Load all P1 files in selected subfolder"):
        #lOADING ALL P1 FILES
           for phase_prefix in ["P1"]:
               # get all files with the given prefix in the selected subfolder
               phase_files = list_files(os.path.join(address, selected_subfolder), phase_prefix)
           for filename in phase_files:
               #st.write(filename)
               file_path = os.path.join(subfolder_path, filename)
               filename = filename[3:-4]
               processed_data[filename] = analyze_data_P1(file_path)
           animal_name = selected_subfolder.replace("_", " ")
           plot_metrics_P1(processed_data, selected_subfolder, animal_name)
        #LOADING ALL P2 FILES
        if st.button("Load all P2 files in selected subfolder"):
        #create a dictionary to store all phase files
           for phase_prefix in ["P2"]:
               # get all files with the given prefix in the selected subfolder
               phase_files = list_files(os.path.join(address, selected_subfolder), phase_prefix)
           for filename in phase_files:
               #st.write(filename)
               file_path = os.path.join(subfolder_path, filename)
               filename = filename[3:-4]
               processed_data[filename] = analyze_data_P2(file_path)
           animal_name = selected_subfolder.replace("_", " ")
           plot_metrics_P2(processed_data, selected_subfolder, animal_name)
        
           
        if st.button("Load all P1 files in all folders"): 
            st.write('Loading all files')
            processed_data = {} # initialize processed_data
            for subfolder in os.listdir(address):
                # check if the item in the parent directory is a subdirectory
                subfolder_path = os.path.join(address, subfolder)
                if os.path.isdir(subfolder_path):
                    # process the files in the subdirectory
                    for phase_prefix in ["P1"]:
                        phase_files = list_files(subfolder_path, phase_prefix)
                        for filename in phase_files:
                            file_path = os.path.join(subfolder_path, filename)
                            if os.path.exists(file_path):
                                filename = filename[3:-4]
                                processed_data[filename] = analyze_data_P1(file_path)

               
            # Extract the animal IDs and names from the processed_data dictionary
            animal_ids = list(set([key.split('_')[0] for key in processed_data.keys()]))
            animal_ids = sorted(animal_ids)
            animal_names = animal_ids
            
            # Plot the data for all animals
            plot_all_metrics_P1(processed_data, animal_ids, animal_names)
            
            #loading all files in all folders Phase2
        if st.button("Load all P2 files in all folders"): 
            st.write('Loading all files')
            processed_data = {} # initialize processed_data
            for subfolder in os.listdir(address):
                # check if the item in the parent directory is a subdirectory
                subfolder_path = os.path.join(address, subfolder)
                if os.path.isdir(subfolder_path):
                    # process the files in the subdirectory
                    for phase_prefix in ["P2"]:
                        phase_files = list_files(subfolder_path, phase_prefix)
                        for filename in phase_files:
                            file_path = os.path.join(subfolder_path, filename)
                            if os.path.exists(file_path):
                                filename = filename[3:-4]
                                processed_data[filename] = analyze_data_P2(file_path)

               
            # Extract the animal IDs and names from the processed_data dictionary
            animal_ids = list(set([key.split('_')[0] for key in processed_data.keys()]))
            animal_ids = sorted(animal_ids)
            animal_names = animal_ids
            
            # Plot the data for all animals
            plot_all_metrics_P2(processed_data, animal_ids, animal_names)
            
            


if __name__ == "__main__":
    main()
