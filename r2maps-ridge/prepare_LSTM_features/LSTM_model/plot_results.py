import math
import sys
import os.path as op
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
# import mne
# from functions import sentcomp_epoching

def regularization_path(model, settings, params):
    fig, ax1 = plt.subplots()

    # Plot regression coef for each regularization size (alpha)
    ax1.plot(model.alphas, model.coefs)
    ax1.set_xscale('log')
    ax1.set_xlabel('Regularization size', size=18)
    ax1.set_ylabel('weights', size=18)
    plt.title(settings.method + ' regression')

    # Plot error on the same figure
    ax2 = ax1.twinx()
    scores = model.cv_results_['mean_test_score']
    scores_std = model.cv_results_['std_test_score']
    std_error = scores_std / np.sqrt(params.CV_fold)
    ax2.plot(model.alphas, scores, 'r.', label='R-squared test set')
    ax2.fill_between(model.alphas, scores + std_error, scores - std_error, alpha=0.2)
    ax2.set_ylabel('R-squared', color='r', size=18)
    ax2.tick_params('y', colors='r')

    scores_train = model.cv_results_['mean_train_score']
    scores_train_std = model.cv_results_['std_train_score']
    std_train_error = scores_train_std / np.sqrt(params.CV_fold)
    ax2.plot(model.alphas, scores_train, 'g.', label='R-squared train set')
    ax2.fill_between(model.alphas, scores_train + std_train_error, scores_train - std_train_error, alpha=0.2)

    plt.axis('tight')
    plt.legend(loc=4)

    return plt


def plot_topomap_optimal_bin(settings, params):

    # Load f-statistic results from Output folder
    f_stats_all = []
    for channel in range(settings.num_MEG_channels):
        file_name = 'MEG_data_sentences_averaged_over_optimal_bin_channel_' + str(channel + 1) + '.npz'
        npzfile = np.load(op.join(settings.path2output, file_name))
        f_stats_all.append(npzfile['arr_1'])

    num_bin_sizes, num_bin_centers = f_stats_all[0].shape

    # Load epochs data from fif file, which includes channel loactions
    epochs = mne.read_epochs(op.join(settings.path2MEGdata, settings.raw_file_name))

    # Generate epochs locked to anomalous words
    anomaly = 0  # 0: normal, 1: nonword (without vowels), 2: syntactic, 3: semantic
    position = [4, 6, 8]  # 0,1,2..8
    responses = [0, 1]  # Correct/wrong response of the subject
    structures = [1, 2, 3]  # 1: 4-4, 2: 2-6, 3: 6-2

    conditions = dict([
        ('Anomalies', [anomaly]),
        ('Positions', position),
        ('Responses', responses),
        ('Structure', structures)])

    knames1, _ = sentcomp_epoching.get_condition(conditions=conditions, epochs=epochs, startTime=-.2,
                                                   duration=1.5, real_speed=params.real_speed/1e3)

    epochs_curr_condition = epochs[knames1]

    # Generate fake power spectrum, to be replace with f-stat later
    freqs = range(51,51 + num_bin_sizes,1); n_cycles = 1
    power = mne.time_frequency.tfr_morlet(epochs_curr_condition, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                          decim=3, n_jobs=4)
    f_stat_object = power[0]
    f_stat_object._data = np.rollaxis(np.dstack(f_stats_all), -1)
    fig_topo = f_stat_object.plot_topo(layout_scale=1.1, title='F-statistic for varying bin sizes and centers', vmin=0, vmax=5,
                            fig_facecolor='w', font_color='k', show=False)


    f_stat_object.times = np.asarray(range(1,num_bin_centers,1))
    f_stat_object.freqs = np.asarray(freqs)

    #subplot_square_side = math.ceil(np.sqrt(num_bin_centers))

    for i, t in enumerate(f_stat_object.times):
        curr_fig = f_stat_object.plot_topomap(tmin=t, tmax=t, fmin=np.min(freqs), fmax=1 + np.min(freqs), show=False)
        file_name = 'f_stats_topomap_patient_' + settings.patient + '_time_' + str(t)
        plt.savefig(op.join(settings.path2figures, file_name))
        plt.close(curr_fig)

    return fig_topo

def plot_topomap_regression_results(settings, params):

    # Load regression results from Output folder
    import pickle
    time_points = range(0, params.SOA + 1, 10)
    best_R_squared_LASSO = np.zeros([306, len(time_points)])
    if settings.collect_data:
        for t, time_point in enumerate(time_points):
            for channel in range(306):#settings.num_MEG_channels):
                pkl_filename = 'Regression_models_' + settings.patient + '_channel_' + str(channel+1) + '_timepoint_' + str(time_point) + '_averaged_over_' + str(params.step) + '_' + settings.LSTM_file_name + '.pckl'
                #'Regression_models_' + settings.patient + '_channel_' + str(channel+1) + '_timepoint_' + str(
                #time_point) + '_' + settings.LSTM_file_name + '.pckl'
                models = []
                # with open(op.join(settings.path2output, pkl_filename), "rb") as f:
                #     while True:
                #         try:
                #             curr_results.append(pickle.load(f))
                #         except EOFError:
                #             break
                with open(op.join(settings.path2output, pkl_filename), "rb") as f:
                    try:
                        curr_results = pickle.load(f)
                        best_R_squared_LASSO[channel, t] = curr_results['lasso_scores_test']
                    except EOFError:
                        break
                    sys.stdout.write('Channel ' + str(channel) + ' time point ' + str(t) + ' ')
                print(best_R_squared_LASSO[channel, t])
                # Extract best score on validation set
                #best_R_squared_LASSO[channel] = models[2].best_score_
                # best_R_squared_LASSO[channel] = models[channel]

		#best_R_squared_LASSO = best_R_squared_LASSO + [0] * 196
		#best_R_squared_LASSO = [[x] for x in best_R_squared_LASSO]

        pkl_filename = 'best_R_squared_Lasso' + settings.patient + '_' + settings.LSTM_file_name + '.pckl'
        with open(op.join(settings.path2output, pkl_filename), 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump(best_R_squared_LASSO, f)
    else:
        pkl_filename = 'best_R_squared_Lasso' + settings.patient + '_' + settings.LSTM_file_name + '.pckl'
        with open(op.join(settings.path2output, pkl_filename), 'r') as f:  # Python 3: open(..., 'wb')
            best_R_squared_LASSO = pickle.load(f)

    # Load epochs data from fif file, which includes channel loactions
    epochs = mne.read_epochs(op.join(settings.path2MEGdata, settings.raw_file_name))

    # Generate epochs locked to anomalous words
    anomaly = 0  # 0: normal, 1: nonword (without vowels), 2: syntactic, 3: semantic
    position = [4, 6, 8]  # 0,1,2..8
    responses = [0, 1]  # Correct/wrong response of the subject
    structures = [1, 2, 3]  # 1: 4-4, 2: 2-6, 3: 6-2

    conditions = dict([
        ('Anomalies', [anomaly]),
        ('Positions', position),
        ('Responses', responses),
        ('Structure', structures)])

    knames1, _ = sentcomp_epoching.get_condition(conditions=conditions, epochs=epochs, startTime=-.2,
                                                   duration=1.5, real_speed=params.real_speed/1e3)

    # Generate fake evoked spectrum, to be replace with regression results later
    evoked = epochs[knames1].average()
    evoked._data = np.asarray(best_R_squared_LASSO)
    evoked.times = np.asarray(time_points)
    #evoked.times = [0]
    #fig_topo = evoked.plot_topomap(times=0, show=True)
    print(time_points)
    fig_topo = evoked.plot_topomap(times=time_points[params.i], show=False)
    fig_topo.axes[1].axes.title._text = 'R-squared'
    fig_topo.axes[0].axes.title._text = str(params.i * 10) + ' msec'
    # ax = plt.gca()  # plt.gca() for current axis, otherwise set appropriately.

    # im = fig_topo.axes[0].images
    # cbar = im[-1].colorbar
    # cbar._values = best_R_squared_LASSO
    # fig_topo.axes[0].images[0].colorbar.vmin = -1
    # fig_topo.axes[0].images[0].colorbar.vmax = 1
    # fig_topo.axes[0].images[0].colorbar.set_ticks([np.min(best_R_squared_LASSO), np.max(best_R_squared_LASSO)])
    fig_topo.axes[0].images[0].colorbar.set_ticklabels(np.linspace(np.min(best_R_squared_LASSO[:, params.i]), np.max(best_R_squared_LASSO[:, params.i]), 5),
                                                       update_ticks=True)
    return fig_topo


def plot_weights(weights, model, axs, i, settings, params):
    fig, ax = plt.subplots()

    ind = np.arange(weights.shape[1]) + 1  # the x locations for the groups
    width = 0.35  # the width of the bars

    axs[i].bar(ind, np.mean(weights, axis=0), width, color='b', yerr=np.std(weights, axis=0))

    ax.set_ylabel('Weight size')
    ax.set_title(model)

    return plt, ax