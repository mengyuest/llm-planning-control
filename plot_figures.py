import os
from os.path import join as ospj
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from collections import defaultdict, Counter
import utils

sns.set_theme(style="whitegrid")
colors = sns.color_palette("tab10")

method_dict = {
    "e2e": "LLM-predict",
    "code": "LLM-code",
    "ours": "LLM-use-API (ours)"
}

scene_dict = {
    "0-track-linear": "Track-lin",
    "1-track-car": "Track-car",
    "2-plan": "Plan-easy",
    "4-hier": "Plan-hard",
    "3-stl": "Plan-STL",
}

scenarios = ["0-track-linear", "1-track-car", "2-plan", "4-hier", "3-stl"]
methods = ["e2e", "code", "ours"]

def plot1(data, img_name, TWO_PLOTS=True):
    STD_SCALE = 0.25
    FONTSIZE = 14

    success_rates = defaultdict(lambda: defaultdict(list))
    rounds_taken = defaultdict(lambda: defaultdict(list))
    for scenario in scenarios:
        for method in methods:
            trials = data[scenario][method]
            for trial_result in trials.values():
                rounds_taken[scenario][method].append(len(trial_result))
                success_rates[scenario][method].append(1 if trial_result[-1]['status']=='success' else 0)
    
    if TWO_PLOTS:
        fig = plt.figure(figsize=(8, 2.5))  # Increased figure width
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 2.5))  # Increased figure width
    bar_width = 0.2
    x = np.arange(len(scenarios))

    # Success rate
    if TWO_PLOTS:
        ax = plt.gca()
    else:
        ax = axes[0]
    for idx, method in enumerate(methods):
        sr_means = [np.clip(np.mean(success_rates[sc][method]), a_min=0.001, a_max=10) for sc in scenarios]
        sr_stds = [np.std(success_rates[sc][method]) * STD_SCALE for sc in scenarios]
        ax.bar(x + idx * bar_width, sr_means, bar_width, yerr=sr_stds, label=method_dict[method], color=colors[idx], capsize=5)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([scene_dict[sce] for sce in scenarios], fontsize=FONTSIZE)
    ax.set_ylim(0, 1)  # Adjusted to 0-1 scale
    ax.set_ylabel('Success rate', fontsize=FONTSIZE)
    if TWO_PLOTS:
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12)
        if ".png" in img_name:
            utils.plt_save_close(img_name.replace(".png", "_1.png"))
        else:
            utils.plt_save_close(img_name.replace(".pdf", "_1.pdf"))

    # Rounds of Query
    if TWO_PLOTS:
        fig = plt.figure(figsize=(8, 2.5))  # Increased figure width
        ax = plt.gca()
    else:
        ax = axes[1]
    for idx, method in enumerate(methods):
        rt_means = [np.mean(rounds_taken[sc][method]) for sc in scenarios]
        rt_stds = [np.std(rounds_taken[sc][method]) * STD_SCALE for sc in scenarios]
        ax.bar(x + idx * bar_width, rt_means, bar_width, yerr=rt_stds, label=method_dict[method] if TWO_PLOTS else None, color=colors[idx], capsize=5)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([scene_dict[sce] for sce in scenarios], fontsize=FONTSIZE)
    ax.set_ylim(0, 6.5)  # Adjusted to 0-1 scale
    ax.set_ylabel('Rounds of query', fontsize=FONTSIZE)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12)

    if TWO_PLOTS:
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12)
        if ".png" in img_name:
            utils.plt_save_close(img_name.replace(".png", "_2.png"))
        else:
            utils.plt_save_close(img_name.replace(".pdf", "_2.pdf"))
    else:
        utils.plt_save_close(img_name)


def plot2(data, img_name):
    FONTSIZE = 14
    success_rates = defaultdict(lambda: defaultdict(list))
    rounds_taken = defaultdict(lambda: defaultdict(list))
    roundwise_success = defaultdict(lambda: defaultdict(lambda: np.zeros(6)))
    
    for scenario in scenarios:
        for method in methods:
            trials = data[scenario][method]
            # for trial_result in trials.values():
            for trial_key, trial_result in trials.items():
                rounds_taken[scenario][method].append(len(trial_result))
                success_rates[scenario][method].append(1 if trial_result[-1]['status']=='success' else 0)
                for round_idx in range(len(trial_result)):
                    status = trial_result[round_idx]['status']
                    if status == "success":
                        roundwise_success[scenario][method][round_idx:] += 1
    
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.))
    rounds = np.arange(1, 7)
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        for m_idx, method in enumerate(methods):
            total_trials = len(success_rates[scenario][method])
            success_curve = roundwise_success[scenario][method] / total_trials
            ax.plot(rounds, success_curve, marker='o', label=method if idx==0 else None, color=colors[m_idx])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(rounds)
        ax.set_title(scene_dict[scenario])
        if idx==2:
            ax.set_xlabel('Rounds of query', fontsize=FONTSIZE)
        if idx==0:
            ax.set_ylabel('Success rate', fontsize=FONTSIZE)
            fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=FONTSIZE-2) 
        else:
            ax.set_yticklabels([])
    utils.plt_save_close(img_name)


def plot3(data, img_name):
    # # Plot 3: LLM Error Pattern Histogram
    FONTSIZE = 14
    methods = ["e2e", "code", "ours"]
    scenarios = ["0-track-linear", "1-track-car", "2-plan", "3-stl", "4-hier"]
    
    # Initialize error patterns counter for each method
    error_patterns = {method: Counter({'success': 0, 'parsing error': 0, 'syntax error': 0, 
                                       'runtime error': 0, 'timeout error': 0, 'failure': 0}) 
                      for method in methods}

    error_code_dict = {
        "success": "Success",
        "parsing error": "Parsing error",
        "syntax error": "Syntax error",
        "runtime error": "Syntax error",
        "timeout error": "Timeout error",
        "failure": "Task failure",
    }

    # Populate error patterns with counts for the final round status
    for scenario in scenarios:
        for method in methods:
            trials = data[scenario][method]
            for trial_result in trials.values():
                final_status = trial_result[-1]['status']  # Consider the final round status
                if final_status in error_patterns[method]:
                    error_patterns[method][final_status] += 1

    # Create pie charts for each method
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    labels = error_code_dict.keys()
    colors = sns.color_palette("Set2", 10)  # Using a Seaborn color palette
    colors2 = sns.color_palette("Paired")
    colors = [colors[idx] for idx in [0, 7]] + [colors2[8], colors2[6], colors2[0], colors2[4]]
    for idx, method in enumerate(methods):
        sub_sizes = [error_patterns[method][status] for status in labels if error_patterns[method][status] > 0]
        sub_readable_labels = [error_code_dict[status] for status in labels if error_patterns[method][status] > 0]
        sub_colors = [colors[s_i] for s_i, status in enumerate(labels) if error_patterns[method][status] > 0]
        
        # Plot pie chart, and hide labels with 0% occurrence
        axes[idx].pie(sub_sizes, labels=sub_readable_labels, colors=sub_colors, 
                      autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'}, 
                      textprops={'fontsize': FONTSIZE-1}, labeldistance=1.05, pctdistance=0.8)
        axes[idx].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        axes[idx].set_title(method_dict[method], fontsize=FONTSIZE)
    
    # Manually create a shared legend for all possible error codes
    legend_labels = [error_code_dict[code] for code in error_code_dict.keys() if code != "syntax error"]
    legend_handles = [Rectangle((0, 0), 1, 1, facecolor=color) for c_i,color in enumerate(colors) if c_i != 2]
    fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.), ncol=6, fontsize=FONTSIZE)
    utils.plt_save_close(img_name)


def plot4(data, img_name):
    # Plot 4: API usage histogram for "ours"
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for idx, scenario in enumerate(scenarios):
        api_counter = Counter()
        trials = data[scenario]["ours"]
        for trial_result in trials.values():
            apis = trial_result[0].get('apis', [])
            api_counter.update(apis)
        api_names, api_counts = zip(*api_counter.most_common())
        sns.barplot(x=list(api_names), y=list(api_counts), ax=axes[idx], palette='Greens_d', hue=list(api_counts))
        axes[idx].set_title(f'API Usage - Scenario: {scenario}')
        axes[idx].set_xticks(axes[idx].get_xticks())
        axes[idx].set_xticklabels(api_names, rotation=45, ha='right')
        axes[idx].set_ylabel('Usage Count')

    fig.suptitle('API Usage Histogram for "Ours" Method')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    utils.plt_save_close(img_name)


def plot_abl1(img_path):
    ###############
    # ablation-0-tau
    ###############
    file_paths1 = {
        "0-track-linear": {
            "tau_0.0": "task0_api_tau.0",
            "tau_0.1": "task0_api",
            "tau_0.2": "task0_api_tau.2",
            "tau_0.3": "task0_api_tau.3",
            "tau_0.5": "task0_api_tau.5",
            "tau_0.7": "task0_api_tau.7",
            "tau_1.0": "task0_api_tau1",
            "tau_1.2": "task0_api_tau1.2",
            "tau_1.5": "task0_api_tau1.5",
            }
    }
    
    all_data1 = {}
    scenarios = ["0-track-linear"]
    for task_type in file_paths1:
        all_data1[task_type]={}
        for tau in file_paths1[task_type]:
            file_path = file_paths1[task_type][tau]
            with open(ospj(EXP_DIR, file_path, "all_metrics.json")) as f:
                json_data = json.load(f)
                all_data1[task_type][tau]=json_data
                print(task_type, tau, len(json_data), 
                      np.mean(["status" in json_data[str(k)][-1] and json_data[str(k)][-1]["status"]=="success"  for k in range(len(json_data))]) )
    
    success_rates = defaultdict(lambda: defaultdict(list))
    rounds_taken = defaultdict(lambda: defaultdict(list))
    for scenario in scenarios:
        for tau in all_data1[scenario]:
            trials = all_data1[scenario][tau]
            for trial_result in trials.values():
                rounds_taken[scenario][tau].append(len(trial_result))
                success_rates[scenario][tau].append(1 if trial_result[-1]['status']=='success' else 0)

    tau_values = [float(tau.split("_")[1]) for tau in success_rates[scenario]]
    success_rate_values = [np.mean(success_rates[scenario][tau]) for tau in success_rates[scenario]]
    rounds_taken_values = [np.mean(rounds_taken[scenario][tau]) for tau in success_rates[scenario]]
        
    FONTSIZE = 14
    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 3))

    # Success rate plot
    ax1.set_xlabel('LLM temperature', fontsize=FONTSIZE)
    ax1.set_ylabel('Success rate', fontsize=FONTSIZE, color='tab:blue')
    ax1.plot(tau_values, success_rate_values, color='tab:blue', linewidth=3.0, markersize=12, marker='o', label='Success rate')
    ax1.axvline(x=0.1, color="gray", linestyle="--", linewidth=3.0)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=FONTSIZE-1)

    # Create a second y-axis for rounds taken
    ax2 = ax1.twinx()
    ax2.set_ylabel('Rounds of query', fontsize=FONTSIZE, color='tab:red')
    ax2.plot(tau_values, rounds_taken_values, color='tab:red', linewidth=3.0, markersize=12, marker='P', label='Rounds of query')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=FONTSIZE-1)
    utils.plt_save_close(ospj(EXP_DIR, img_path))


def plot_abl2(img_path):
    methods = ["gpt-4o-mini", "gpt-4o", "gpt-o1"]
    success_rates = [0.45, 0.40, 0.77]
    runtimes = [(1*3600+24*60+33)/100, (2*3600+14*60+21)/100, (3*3600+58*60+5)/100,]
    methods.append("gpt-o3-mini")
    success_rates.append(0.26)
    runtimes.append((1*3600+3*60+27)/23)
    methods = [xxx.replace("gpt-", "") for xxx in methods]
    
    # Set up the figure and axes for two bar plots
    FONTSIZE = 14
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Bar plot for success rates
    axes[0].bar(methods, success_rates, color='skyblue')
    axes[0].set_ylabel('Success rate', fontsize=FONTSIZE)
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='both', labelsize=FONTSIZE-1)
    
    # Bar plot for runtimes
    axes[1].bar(methods, runtimes, color='salmon')
    axes[1].set_ylabel('Average runtime (sec)', fontsize=FONTSIZE)
    axes[1].set_ylim(0, max(runtimes) + 1)
    axes[1].tick_params(axis='both', labelsize=FONTSIZE-1)

    # Tight layout and show the plot
    fig.tight_layout()
    utils.plt_save_close(ospj(EXP_DIR, img_path))


def plot_abl3(img_path, TWO_PLOTS=True, STACKED=False):
    file_paths = {
        "0-track-linear": {
                "ours": "task0_api",
                "ours_all": "task0_api_all",
            },
        "1-track-car": {
                "ours": "task0_api_uni",
                "ours_all": "task0_api_uni_all",
            },
        "2-plan": {
                "ours": "task1_api",
                "ours_all": "task1_api_all",
            },
        "3-stl": {
                "ours": "task4_api",
                "ours_all": "task4_api",
            },
        "4-hier": {
                "ours": "task3_api",
                "ours_all": "task3_api_all",
            }
    }
    
    FONTSIZE = 14
    
    # Function to load data from file
    def load_data(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    # Initialize a dictionary to store the metrics for each scenario
    all_data = {}

    # Load the data for each scenario
    for scenario in file_paths:
        all_data[scenario] = {}
        for method in file_paths[scenario]:
            file_path = file_paths[scenario][method]
            data = load_data(os.path.join(EXP_DIR, file_path, "all_metrics.json"))
            all_data[scenario][method] = data

    # Extract success rates and rounds taken for each scenario and method
    success_rates = defaultdict(lambda: {})
    rounds_taken = defaultdict(lambda: {})

    method_dict = {
    "e2e": "LLM-predict",
    "code": "LLM-code",
    "ours": "LLM-use-API (ours)",
    "ours_all": "LLM-use-API (all)"
    }

    scene_dict = {
        "0-track-linear": "Track-lin",
        "1-track-car": "Track-car",
        "2-plan": "Plan-easy",
        "4-hier": "Plan-hard",
        "3-stl": "Plan-STL",
    }
    
    for scenario in all_data:
        for method in all_data[scenario]:
            trials = all_data[scenario][method]
            success_rate = np.mean([1 if trial[-1]["status"] == "success" else 0 for trial in trials.values()])
            rounds = np.mean([len(trial) for trial in trials.values()])
            success_rates[scenario][method] = success_rate
            rounds_taken[scenario][method] = rounds

    # Prepare the data for bar plot
    scenarios = list(file_paths.keys())

    # Success rates and rounds data for plotting
    success_rate_values = [success_rates[scenario]["ours"] for scenario in scenarios]
    ours_all_success_rate_values = [success_rates[scenario]["ours_all"] for scenario in scenarios]
    rounds_values = [rounds_taken[scenario]["ours"] for scenario in scenarios]
    ours_all_rounds_values = [rounds_taken[scenario]["ours_all"] for scenario in scenarios]

    # Create the bar plot
    if TWO_PLOTS:
        fig = plt.figure(figsize=(8, 2.5))
        ax = plt.gca()
    else:
        if STACKED:
            fig, axes = plt.subplots(2, 1, figsize=(8, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 2.5))

    # Bar plot for Success Rates
    if TWO_PLOTS:
        ax = plt.gca()
    else:
        ax = axes[0]
    x = np.arange(len(scenarios))
    bar_width = 0.3
    ax.bar(x - bar_width / 2, success_rate_values, bar_width, label=method_dict["ours"], color=colors[2])  #color='skyblue')
    ax.bar(x + bar_width / 2, ours_all_success_rate_values, bar_width, label=method_dict["ours_all"], color=colors[3])  #color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels([scene_dict[ss] for ss in scenarios], fontsize=FONTSIZE) # , rotation=45)
    ax.set_ylabel("Success rate", fontsize=FONTSIZE)
    
    if TWO_PLOTS:
        utils.plt_save_close(img_path.replace(".pdf", "_1.pdf").replace(".png", "_1.png"))
        fig = plt.figure(figsize=(7.5, 3))
        ax = plt.gca()
    else:
        ax = axes[1]
    # Bar plot for Rounds Taken
    ax.bar(x - bar_width / 2, rounds_values, bar_width, label=method_dict["ours"], color=colors[2])  #color='skyblue')
    ax.bar(x + bar_width / 2, ours_all_rounds_values, bar_width, label=method_dict["ours_all"], color=colors[3])  #color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels([scene_dict[ss] for ss in scenarios], fontsize=FONTSIZE)  # , rotation=45)
    ax.set_ylabel("Rounds of query", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE-2, ncol=2)

    if TWO_PLOTS:
        utils.plt_save_close(img_path.replace(".pdf", "_2.pdf").replace(".png", "_2.png"))
    else:
        if STACKED:
            utils.plt_save_close(img_path.replace(".pdf", "_stack.pdf").replace(".png", "_stack.png"))
        else:
            utils.plt_save_close(img_path)

def main():
    file_paths = {
        "0-track-linear": {
                "e2e": "task0_e2e",
                "code": "task0_code",
                "ours": "task0_api",
            },
        "1-track-car": {
                "e2e": "task0_e2e_uni",
                "code": "task0_code_uni",
                "ours": "task0_api_uni",
            },
        "2-plan": {
                "e2e": "task1_e2e",
                "code": "task1_code",
                "ours": "task1_api",
            },
        "3-stl": {
                "e2e": "task4_e2e",
                "code": "task4_code",
                "ours": "task4_api",
            },
        "4-hier": {
                "e2e": "task3_e2e",
                "code": "task3_code",
                "ours": "task3_api",
            }
    }
    
    # load data
    all_data = {}
    for task_type in file_paths:
        all_data[task_type]={}
        for method in file_paths[task_type]:
            file_path = file_paths[task_type][method]
            with open(ospj(EXP_DIR, file_path, "all_metrics.json")) as f:
                json_data = json.load(f)
                all_data[task_type][method]=json_data
                print(task_type, method, len(json_data), np.mean(["status" in json_data[str(k)][-1] and json_data[str(k)][-1]["status"]=="success"  for k in range(len(json_data))]) )

    # FIG.1 report average success rate (with standard deviation) and average rounds used (with standard deviation)
    # FIG.2 report how multi-round feedbacks help increase accuracy
    # FIG.3 report LLM_error distribution
    # FIG.4 visualization
    plot1(all_data, img_name=ospj(EXP_DIR, "figure_1.pdf"), TWO_PLOTS=True)
    plot2(all_data, img_name=ospj(EXP_DIR, "figure_2.pdf"))
    plot3(all_data, img_name=ospj(EXP_DIR, "figure_3.pdf"))
    plot4(all_data, img_name=ospj(EXP_DIR, "figure_4.pdf"))
    
    # FIG.5 Ablation on LLM temperature
    # FIG.6 Ablation on different LLM models
    # FIG.7 Ablation on use all code or select-then-code
    plot_abl1(img_path=ospj(EXP_DIR, "figure_abl_1.pdf"))
    plot_abl2(img_path=ospj(EXP_DIR, "figure_abl_2.pdf"))
    plot_abl3(ospj(EXP_DIR, "figure_abl_3_ours.pdf"), TWO_PLOTS=True)
    
if __name__ == "__main__":
    EXP_DIR = utils.get_exp_dir()
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2-t1))