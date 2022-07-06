import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps
from sklearn.svm import SVR, NuSVR, NuSVC
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from scipy import stats as sc_stats

# from sklearn.neighbors import DistanceMetric
# from sklearn.cluster import KMeans

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os
import shutil
import posixpath
import time
from timeit import default_timer as timer

# from pynput import mouse

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.append("/home/pdp1145/fetal_ecg_det/wfdb_python_master/")

# import kymatio

import wfdb
import pywt

n_taps = 101
f = 0.015
fir_hipass = sps.firwin(n_taps, f, pass_zero=False)

trimmed_mean_wdw_size = 256
trim_fac = 0.25

# n_taps = 11
# f_hipass = 5.0
# fir_hipass = sps.firwin(n_taps, f_hipass, width=None, window='hamming', pass_zero='highpass', scale=False, nyq=None, fs=1000.0)

# Show high pass filter & freq resp:
#
x_idxs = np.arange(n_taps)
figz = make_subplots(rows=2, cols=1, subplot_titles=("FIR Coefs", "Freq Resp"))
figz.append_trace(go.Scatter(x=x_idxs, y=fir_hipass), row=1, col=1)
# figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead[init_record_skip: (init_record_skip + overlap_wdw_idx)]), row=2, col=1)
figz.show()

# record = wfdb.rdrecord('/home/pdp1145/fetal_ecg_det/wfdb_python_master/sample-data/a103l')
# record = wfdb.rdrecord('/home/pdp1145/fetal_ecg_det/fetal_ecg_data/ARR_01')
record = wfdb.rdrecord('/home/pdp1145/fetal_ecg_det/fetal_ecg_data/NR_09')
# wfdb.plot_wfdb(record=record, title='Record a103l from Physionet Challenge 2015')

# record = wfdb.rdrecord('/home/pdp1145/fetal_ecg_det/fetal_ecg_data_set_a/set-a/a03')
# ann = wfdb.rdann('/home/pdp1145/fetal_ecg_det/fetal_ecg_data_set_a/set-a/a03', 'fqrs')


rem_record_lth = np.int(record.sig_len/400)
n_bpfs = 4   # 64
scale_fac = 8.0  # 256/n_bpfs
cwt_wdw_lth_h = 1  # 16
svr_wdw_lth = 128
svr_wdw_lth_inv = 1.0/float(svr_wdw_lth)
n_coef_tpls = 1000
k_nn = 10
abdominal_est_outlier_rem_fac = 0.25   # Percentage of abdominal estimates to remove as potential outliers
template_update_fac = 0.1               # Update rate for the best match to current maternal/fetal feature vector
init_record_skip = 1000
wdw_shift = 1
plot_freq = 200

mat_lead_r = np.float32(record.p_signal[0:rem_record_lth,0])
fetal_lead_r = np.float32(record.p_signal[0:rem_record_lth,2])

# mat_lead = sps.convolve(mat_lead_r, fir_hipass, mode='same', method='direct')
# fetal_lead = sps.convolve(fetal_lead_r, fir_hipass, mode='same', method='direct')

# Trimmed mean filtering:
#
mat_lead_med = np.zeros((rem_record_lth,))
fetal_lead_med = np.zeros((rem_record_lth,))
mat_lead = np.zeros((rem_record_lth,))
fetal_lead = np.zeros((rem_record_lth,))

trim_wdw_offset = int(trimmed_mean_wdw_size/2)
for i in np.arange(0, rem_record_lth - trimmed_mean_wdw_size):
    mat_lead_med[i+trim_wdw_offset] = sc_stats.trim_mean(mat_lead_r[i : i + trimmed_mean_wdw_size], trim_fac)
    fetal_lead_med[i+trim_wdw_offset] = sc_stats.trim_mean(fetal_lead_r[i : i + trimmed_mean_wdw_size], trim_fac)

mat_lead = np.subtract(mat_lead_r, mat_lead_med)
fetal_lead = np.subtract(fetal_lead_r, fetal_lead_med)

x = np.arange(len(mat_lead))
fig = make_subplots(rows=2, cols=1, subplot_titles=("Raw Maternal Lead", "Filtered Maternal Lead"))
fig.append_trace(go.Scatter(x=x, y=mat_lead_r), row=1, col=1)
fig.append_trace(go.Scatter(x=x, y=mat_lead), row=2, col=1)
fig.show()

fig = make_subplots(rows=2, cols=1, subplot_titles=("Raw Fetal Lead", "Filtered Fetal Lead"))
fig.append_trace(go.Scatter(x=x, y=fetal_lead_r), row=1, col=1)
fig.append_trace(go.Scatter(x=x, y=fetal_lead), row=2, col=1)
fig.show()

fig = make_subplots(rows=2, cols=1, subplot_titles=("Filtered Maternal Lead", "Filtered Fetal Lead"))
fig.append_trace(go.Scatter(x=x, y=mat_lead), row=1, col=1)
fig.append_trace(go.Scatter(x=x, y=fetal_lead), row=2, col=1)
fig.show()

widths = np.arange(1, (n_bpfs+1))*scale_fac

cwt_maternal_lead = np.float32(sps.cwt(mat_lead, sps.ricker, widths))
cwt_fetal_lead = np.float32(sps.cwt(fetal_lead, sps.ricker, widths))

cwt_trans = np.float32(np.transpose(cwt_maternal_lead))
cwt_trans_fetal = np.float32(np.transpose(cwt_fetal_lead))

fig_cwt_mat, ax_cwt_mat = plt.subplots()
ax_cwt_mat.imshow(cwt_maternal_lead[:, 0 : 1000], aspect='auto')
time.sleep(5.0)
fig_cwt_fetal, ax_cwt_fetal = plt.subplots()
ax_cwt_fetal.imshow(cwt_fetal_lead[:, 0 : 1000], aspect='auto')
# ax_cwt_fetal.imshow(cwt_trans_fetal[:, 0 : 1000], aspect='auto')
time.sleep(5.0)

# SVR w/ single CWT vector -> fetal ECG:
#
n_feats = (cwt_wdw_lth_h*2 -1)*n_bpfs
n_feats_maternal = n_feats*svr_wdw_lth
n_feats_maternal_fetal = n_feats*2*svr_wdw_lth

maternal_feature_vectors = np.float32(np.zeros([n_coef_tpls, n_feats_maternal]))
maternal_fetal_feature_vectors = np.float32(np.zeros([n_coef_tpls, n_feats_maternal_fetal]))
linear_regression_coefs = np.float32(np.zeros([n_coef_tpls, n_feats]))
linear_regression_intercepts = np.float32(np.zeros([n_coef_tpls,]))
mat_lead_wdw_hist = np.float32(np.zeros([n_coef_tpls, svr_wdw_lth]))

n_maternal_fetal_feature_vectors = 0
init_delay = init_record_skip

mat_lead_wdw_hist_arr = np.float32(np.zeros(rem_record_lth,))
abdominal_est = np.float32(np.zeros(rem_record_lth,))
abdominal_est_idxs = np.arange(0, n_coef_tpls)
dist_arr = np.float32(np.zeros(rem_record_lth,))
n_svrs = 0
overlap_wdw_idx = 0
init = 0

if(init == 1):  # Load initialized template library and regressors if already initialized
    maternal_fetal_feature_vectors = np.load('maternal_fetal_feature_vectors1k.npy')
    maternal_feature_vectors = np.float32(np.load('maternal_feature_vectors1k.npy'))
    linear_regression_coefs = np.float32(np.load('linear_regression_coefs1k.npy'))
    linear_regression_intercepts = np.float32(np.load('linear_regression_intercepts1k.npy'))

    n_svrs = n_coef_tpls*wdw_shift                # Skip past template library initialization
    init_delay = init_delay + n_svrs
    overlap_wdw_idx = overlap_wdw_idx + n_svrs


for svr_wdw_beg in np.arange(init_delay, init_delay + rem_record_lth - svr_wdw_lth -1, wdw_shift):

        init_sect_beg = timer()

        wdw_beg = svr_wdw_beg
        wdw_end = wdw_beg + svr_wdw_lth
        fetal_lead_wdw = np.float32(np.zeros([(wdw_end - wdw_beg),]))
        mat_lead_wdw = np.float32(np.zeros([(wdw_end - wdw_beg),]))
        cwt_wdw = np.float32(np.zeros([(wdw_end - wdw_beg), n_feats]))
        cwt_wdw_fetal = np.float32(np.zeros([(wdw_end - wdw_beg), n_feats]))
        regr_idx = 0
        init_sect_end = timer()
        # print(" Init sect elapsed time:  @  " + str(svr_wdw_beg) + "      "   +  str(init_sect_end - init_sect_beg))
        init_sect_beg = timer()

        for wdw_idx in np.arange(wdw_beg, wdw_end):

            fetal_lead_wdw[regr_idx] = fetal_lead[wdw_idx]    # Extract lead windows for regression
            mat_lead_wdw[regr_idx] = mat_lead[wdw_idx]

            if(cwt_wdw_lth_h > 1):
                cwt_wdw[regr_idx,:] = cwt_trans[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h -1, :].flatten()     # Extract feature vectors for regression & knn
                cwt_wdw_fetal[regr_idx,:] = cwt_trans_fetal[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h -1, :].flatten()     # Extract feature vectors for regression & knn
            else:
                cwt_wdw[regr_idx,:] = cwt_trans[wdw_idx, :].flatten()     # Extract feature vectors for regression & knn
                cwt_wdw_fetal[regr_idx,:] = cwt_trans_fetal[wdw_idx, :].flatten()     # Extract feature vectors for regression & knn


            # Overwrite fetal lead with one of the CWT channels:
            fetal_lead_wdw[regr_idx] = cwt_trans_fetal[wdw_idx, 0]
            regr_idx = regr_idx +1

        # Overwrite cwt_wdw[] data w/ more distinct signals:
        xt = np.arange(0, regr_idx)
        cwt_wdw[:, 0] = np.sin(xt*0.12)
        cwt_wdw[:, 1] = np.sin(xt*0.1)
        cwt_wdw[:, 2] = -np.sin(xt*0.08)

        rng = np.random.default_rng(seed=42)
        cwt_wdw[:, 3] = rng.random((regr_idx))*1000.0
        # cwt_wdw[:, 3] = -np.sin(xt*0.06)

        # fetal_lead_wdw = cwt_wdw[:, 3]
        fetal_lead_wdw = np.sin(xt*0.025)

        init_sect_end = timer()
        # print(" Array collection sect elapsed time:  @  " + str(svr_wdw_beg) + "      "   +  str(init_sect_end - init_sect_beg))

        if(n_svrs < n_coef_tpls):       # Initialization phase (fill template library)

            init_sect_beg = timer()

            # Save maternal feature vectors & composite maternal / fetal feature vectors:
            maternal_feature_vectors[n_svrs, :] = cwt_wdw.flatten()
            maternal_fetal_feature_vectors[n_svrs, :] = np.concatenate((cwt_wdw.flatten(), cwt_wdw_fetal.flatten()), axis = None)

            # Show inputs and target:
            #
            x = np.arange(len(fetal_lead_wdw))
            fig = make_subplots(rows=5, cols=1, subplot_titles=("CWT 1", "CWT 2", "CWT 3", "CWT 4", "Target"))
            fig.append_trace(go.Scatter(x=x, y=cwt_wdw[:, 0]), row=1, col=1)
            fig.append_trace(go.Scatter(x=x, y=cwt_wdw[:, 1]), row=2, col=1)
            fig.append_trace(go.Scatter(x=x, y=cwt_wdw[:, 2]), row=3, col=1)
            fig.append_trace(go.Scatter(x=x, y=cwt_wdw[:, 3]), row=4, col=1)
            fig.append_trace(go.Scatter(x=x, y=fetal_lead_wdw), row=5, col=1)
            fig.show()

            #
            # Linear support vector regression:
            #   - Use for RFE to distill features (impulse response families) for subsequent input to polynomial SVR
            #   - Scale features using linear SVR coef's prior to input to poly SVR
            #
            #
            # nusv_res = NuSVR(nu=0.95, C=10.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=10000)
            nusv_res = NuSVR(nu=0.5, C=1.0, kernel='linear', degree=2, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
            z_rbf = nusv_res.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)


            # Store regression coef's & offset:
            nusv_lin_coef = np.float32(nusv_res.coef_)
            nusv_intercept = np.float32(nusv_res.intercept_)

            linear_regression_coefs[n_svrs, :] = nusv_lin_coef
            linear_regression_intercepts[n_svrs] = nusv_intercept

            mat_lead_wdw_hist[n_svrs, :] = mat_lead_wdw         # Save maternal lead for this window (debug only)

            # Show fetal lead & predicted fetal lead (from maternal lead):
            #
            if(cwt_wdw_lth_h > 1):
                sv_coef_unwrapped = np.reshape(nusv_lin_coef,[(2*cwt_wdw_lth_h -1),n_bpfs])
            else:
                sv_coef_unwrapped = nusv_lin_coef
            sv_coef_sf = 1.0/np.max(nusv_lin_coef)
            sv_coef_norm = nusv_lin_coef*sv_coef_sf
            x = np.arange(len(fetal_lead_wdw))
            fig = make_subplots(rows=3, cols=1, subplot_titles=("Raw Maternal Lead", "Raw Fetal Lead", "Predicted Fetal Lead"))
            fig.append_trace(go.Scatter(x=x, y=mat_lead_wdw), row=1, col=1)
            fig.append_trace(go.Scatter(x=x, y=fetal_lead_wdw), row=2, col=1)
            fig.append_trace(go.Scatter(x=x, y=z_rbf), row=3, col=1)
            # fig.append_trace(go.Scatter(x=x, y=sv_coef_norm), row=4, col=1)
            fig.show()

            # Unwrapped linear SV coef's:
            # fig_cwt_mat, ax_cwt_mat = plt.subplots()
            # ax_cwt_mat.imshow(sv_coef_unwrapped, aspect='auto')

            x = np.arange((sv_coef_unwrapped.size))
            fig = make_subplots(rows=1, cols=1, subplot_titles=("Linear Coefs"))
            fig.append_trace(go.Scatter(x=x, y=sv_coef_unwrapped[0,:]), row=1, col=1)
            fig.show()

            #
            # Polynomial support vector regression:
            #   - Use for consideration of higher order interactions between selected, hypothesized impulse responses
            #   - Scaled features (impulses responses) allow optimization of internal non-linear SVR feature space
            #
            nusv_res = NuSVR(nu=0.5, C=1.0, kernel='poly', degree=2, gamma=1.001, coef0=1.0, shrinking=True, tol=0.001,
                             cache_size=200, verbose=False, max_iter=-1)
            z_rbf = nusv_res.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)

            # Show fetal lead & predicted fetal lead (from maternal lead):
            #
            x = np.arange(len(fetal_lead_wdw))
            fig = make_subplots(rows=3, cols=1,
                                subplot_titles=("Poly Raw", "Raw Fetal Lead", "Predicted Fetal Lead"))
            fig.append_trace(go.Scatter(x=x, y=mat_lead_wdw), row=1, col=1)
            fig.append_trace(go.Scatter(x=x, y=fetal_lead_wdw), row=2, col=1)
            fig.append_trace(go.Scatter(x=x, y=z_rbf), row=3, col=1)
            # fig.append_trace(go.Scatter(x=x, y=sv_coef_norm), row=4, col=1)
            fig.show()

            # Pre-scale features prior to poly SVR:
            #
            for jj in np.arange(0, n_bpfs):
                cwt_wdw[:,jj] = cwt_wdw[:,jj]*sv_coef_unwrapped[0,jj]

            # cwt_wdw[:, 3] = 0.0;
            #
            # Polynomial support vector regression:
            #   - Use for consideration of higher order interactions between selected, hypothesized impulse responses
            #   - Scaled features (impulses responses) allow optimization of internal non-linear SVR feature space
            #
            nusv_res = NuSVR(nu=0.5, C=100.0, kernel='poly', degree=2, gamma=1.001, coef0=1.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
            z_rbf = nusv_res.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)

            # Show fetal lead & predicted fetal lead (from maternal lead):
            #
            x = np.arange(len(fetal_lead_wdw))
            fig = make_subplots(rows=3, cols=1, subplot_titles=("Poly Scaled", "Raw Fetal Lead", "Predicted Fetal Lead"))
            fig.append_trace(go.Scatter(x=x, y=mat_lead_wdw), row=1, col=1)
            fig.append_trace(go.Scatter(x=x, y=fetal_lead_wdw), row=2, col=1)
            fig.append_trace(go.Scatter(x=x, y=z_rbf), row=3, col=1)
            # fig.append_trace(go.Scatter(x=x, y=sv_coef_norm), row=4, col=1)
            fig.show()

            # Unwrapped linear SV coef's:
            # fig_cwt_mat, ax_cwt_mat = plt.subplots()
            # ax_cwt_mat.imshow(sv_coef_unwrapped, aspect='auto')

            x = np.arange((sv_coef_unwrapped.size))
            fig = make_subplots(rows=1, cols=1, subplot_titles=("Linear Coefs"))
            fig.append_trace(go.Scatter(x=x, y=sv_coef_unwrapped[0,:]), row=1, col=1)
            fig.show()




            # it_sect_end = timer()
            # print(" NuSVR sect elapsed time:  @  " + str(svr_wdw_beg) + "      " + str(init_sect_end - init_sect_beg))

            # Generate abdominal signal estimate for this window:
            #   - initialization region only
            cwt_wdw_trans = np.transpose(cwt_wdw)
            z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
            z_cwt_xcoef_rs = (np.reshape(z_cwt_xcoef, (svr_wdw_lth,)) + nusv_intercept) * svr_wdw_lth_inv

            abdominal_est[(init_record_skip + overlap_wdw_idx): (init_record_skip + overlap_wdw_idx + svr_wdw_lth)] = \
                np.add(z_cwt_xcoef_rs, abdominal_est[(init_record_skip + overlap_wdw_idx): ( init_record_skip + overlap_wdw_idx + svr_wdw_lth)])

        #        abdominal_est[init_record_skip + overlap_wdw_idx + int(svr_wdw_lth/2)] = \
        #                                np.add(z_cwt_xcoef_rs[int(svr_wdw_lth/2)], abdominal_est[init_record_skip + overlap_wdw_idx + int(svr_wdw_lth/2)])

        else:       # Estimates based on retrieved templates -> update templates

            # Maternal CWT templates centered on this sample:
            maternal_feature_vector_s = cwt_wdw.flatten()
            maternal_feature_vector_rs = np.reshape(maternal_feature_vector_s, (1, maternal_feature_vector_s.size))

            # Maternal & fetal CWT templates centered on this sample:
            maternal_fetal_feature_vector_s = np.concatenate((cwt_wdw.flatten(), cwt_wdw_fetal.flatten()), axis=None)
            maternal_fetal_feature_vector_rs = np.reshape(maternal_fetal_feature_vector_s, (1, maternal_fetal_feature_vector_s.size))

            # Get k-nn maternal lead templates:
            token_dists_knn = distance.cdist(maternal_feature_vector_rs, maternal_feature_vectors, metric='cityblock')
            token_dists_knn_sorted_idxs = np.argsort(token_dists_knn).flatten()

            # Sorted distances (for debug only for now):
            token_dists_knn_fl = token_dists_knn.flatten()
            token_dists_knn_sorted = token_dists_knn_fl[token_dists_knn_sorted_idxs]
            # dist_arr[init_record_skip + overlap_wdw_idx + int(svr_wdw_lth / 2)] = token_dists_knn_sorted[0]

            # Regenerate maternal lead from best matches to maternal lead: (debug only)
            mat_wdw_knn = mat_lead_wdw_hist[token_dists_knn_sorted_idxs[0],:]
            mat_lead_wdw_hist_arr[(init_record_skip + overlap_wdw_idx): (init_record_skip + overlap_wdw_idx + svr_wdw_lth)] = \
                np.add(mat_wdw_knn, mat_lead_wdw_hist_arr[(init_record_skip + overlap_wdw_idx): (init_record_skip + overlap_wdw_idx + svr_wdw_lth)])

            # Show sorted distances:
            #
            # token_dist_knn_idxs = np.arange(len(token_dists_knn_sorted))
            # fig = make_subplots(rows=1, cols=1)
            # fig.append_trace(go.Scatter(x=token_dist_knn_idxs, y=token_dists_knn_sorted), row=1, col=1)
            # fig.show()

            # Retrieve regression coef's from best matches:
            #
            k_fac_inv = 1.0  #  / float(k_nn)
            n_core_ests = int((1.0 - abdominal_est_outlier_rem_fac)*k_nn)
            # n_core_ests_inv = 1.0/float (n_core_ests)
            overlap_fac = svr_wdw_lth_inv * k_fac_inv
            z_cwt_xcoef_rs_sum = np.zeros((svr_wdw_lth,))
            z_cwt_xcoef_rs_mtx = np.zeros((k_nn, svr_wdw_lth))
            cwt_wdw_trans = np.transpose(cwt_wdw)

            for ki in np.arange(0, k_nn):   # Retrieve and sum regression from closest k maternal lead templates:

                nusv_lin_coef = linear_regression_coefs[token_dists_knn_sorted_idxs[ki], :]
                nusv_intercept = linear_regression_intercepts[token_dists_knn_sorted_idxs[ki]]

                # Generate abdominal signal estimate for this window:
                #
                z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
                z_cwt_xcoef_rs = np.reshape(z_cwt_xcoef, (svr_wdw_lth,)) + nusv_intercept
                z_cwt_xcoef_rs_sum = z_cwt_xcoef_rs_sum + z_cwt_xcoef_rs
                z_cwt_xcoef_rs_mtx[ki,:] = z_cwt_xcoef_rs

            # Trim outliers from set of estimates for this window:
            #
            z_cwt_xcoef_rs_mean = z_cwt_xcoef_rs_sum/float(k_nn)

            # Rank order window estimates using distance to mean window estimate:
            #
            z_cwt_xcoef_rs_mean = np.reshape(z_cwt_xcoef_rs_mean, (1,svr_wdw_lth))
            wdw_est_dists = distance.cdist(z_cwt_xcoef_rs_mean, z_cwt_xcoef_rs_mtx, metric='euclidean')

            wdw_est_dists_sorted_idxs = np.argsort(wdw_est_dists).flatten()
            wdw_est_dists_fl = wdw_est_dists.flatten()
            wdw_est_dists_sorted = wdw_est_dists_fl[wdw_est_dists_sorted_idxs]

            # Mean abdominal estimate for this window w/ outliers removed:
            wdw_est_rmean = np.mean(z_cwt_xcoef_rs_mtx[wdw_est_dists_sorted_idxs[0 : n_core_ests],:], axis=0)*svr_wdw_lth_inv

            abdominal_est[(init_record_skip + overlap_wdw_idx): (init_record_skip + overlap_wdw_idx + svr_wdw_lth)] = \
                    np.add(wdw_est_rmean, abdominal_est[(init_record_skip + overlap_wdw_idx): (init_record_skip + overlap_wdw_idx + svr_wdw_lth)])

            # abdominal_est[(init_record_skip + overlap_wdw_idx): (init_record_skip + overlap_wdw_idx + svr_wdw_lth)] = \
            #         np.add(z_cwt_xcoef_rs_sum, abdominal_est[(init_record_skip + overlap_wdw_idx): (init_record_skip + overlap_wdw_idx + svr_wdw_lth)])


            #
            #
            # Now find closest maternal / fetal feature vector for this window & update template:
            #

            # Get k-nn maternal/fetal lead templates:
            token_dists_knn = distance.cdist(maternal_fetal_feature_vector_rs, maternal_fetal_feature_vectors, metric='euclidean')
            token_dists_knn_sorted_idxs = np.argsort(token_dists_knn).flatten()

            # Update closest maternal / fetal template:
            maternal_fetal_feature_vectors[token_dists_knn_sorted_idxs[0],:] = \
                maternal_fetal_feature_vectors[token_dists_knn_sorted_idxs[0],:]*(1.0 - template_update_fac) + maternal_fetal_feature_vector_rs*template_update_fac

            # Sorted distances (for debug only for now):
            token_dists_knn_fl = token_dists_knn.flatten()
            token_dists_knn_sorted = token_dists_knn_fl[token_dists_knn_sorted_idxs]
            dist_arr[init_record_skip + overlap_wdw_idx + int(svr_wdw_lth / 2)] = token_dists_knn_sorted[0]



#         # Generate abdominal signal estimate for this window:
#         #
#         cwt_wdw_trans = np.transpose(cwt_wdw)
#         z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
#         z_cwt_xcoef_rs = (np.reshape(z_cwt_xcoef, (svr_wdw_lth,)) + nusv_intercept)*svr_wdw_lth_inv
#
#         abdominal_est[(init_record_skip + overlap_wdw_idx) : (init_record_skip + overlap_wdw_idx + svr_wdw_lth)] = \
#                               np.add(z_cwt_xcoef_rs, abdominal_est[(init_record_skip + overlap_wdw_idx) : (init_record_skip + overlap_wdw_idx + svr_wdw_lth)])
# #        abdominal_est[init_record_skip + overlap_wdw_idx + int(svr_wdw_lth/2)] = \
# #                                np.add(z_cwt_xcoef_rs[int(svr_wdw_lth/2)], abdominal_est[init_record_skip + overlap_wdw_idx + int(svr_wdw_lth/2)])
      
        overlap_wdw_idx = overlap_wdw_idx +1
        n_svrs = n_svrs +1


        if((n_svrs % 50) == 1214):
            figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal",
                    "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4, Training Record Length = 5000", "Abdominal Estimate"))
            figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
            figz.append_trace(go.Scatter(x = x_idxs, y = fetal_lead_wdw), row=2, col=1)
            figz.append_trace(go.Scatter(x = x_idxs, y = z_cwt_xcoef_rs), row=3, col=1)
            figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=3, col=1)
            figz.show()
            time.sleep(5.0)

        if ((n_svrs % plot_freq) == 0):
            x_idxs = np.arange(n_svrs - plot_freq, n_svrs)
            figz = make_subplots(rows=4, cols=1, subplot_titles=("Maternal", "Abdominal", "Abdominal Estimate"))
            figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead[n_svrs - plot_freq : n_svrs]), row=1, col=1)
            figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead[n_svrs - plot_freq : n_svrs]), row=2, col=1)
            figz.append_trace(go.Scatter(x=x_idxs, y=abdominal_est[n_svrs - plot_freq : n_svrs]), row=3, col=1)

            abdominal_nmaternal = np.subtract(fetal_lead, abdominal_est)
            figz.append_trace(go.Scatter(x=x_idxs, y=abdominal_nmaternal[n_svrs - plot_freq : n_svrs]), row=4, col=1)
            # figz.append_trace(go.Scatter(x=x_idxs, y=dist_arr[n_svrs - plot_freq : n_svrs]), row=4, col=1)

            # x_idxs = np.arange(init_record_skip, (init_record_skip + overlap_wdw_idx))
            # figz = make_subplots(rows=4, cols=1, subplot_titles=("Maternal", "Abdominal", "Abdominal Estimate"))
            # figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead[init_record_skip : (init_record_skip + overlap_wdw_idx)]), row=1, col=1)
            # figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead[init_record_skip : (init_record_skip + overlap_wdw_idx)]), row=2, col=1)
            # figz.append_trace(go.Scatter(x=x_idxs, y=abdominal_est[init_record_skip : (init_record_skip + overlap_wdw_idx)]), row=3, col=1)
            # figz.append_trace(go.Scatter(x=x_idxs, y=dist_arr[init_record_skip : (init_record_skip + overlap_wdw_idx)]), row=4, col=1)
            # figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead_wdw_hist_arr[init_record_skip : (init_record_skip + overlap_wdw_idx)]), row=4, col=1)
            
            figz.show()
            time.sleep(10.0)

            if(init == 0):
                np.save('maternal_fetal_feature_vectors1k', maternal_fetal_feature_vectors, allow_pickle=False)
                np.save('maternal_feature_vectors1k', maternal_feature_vectors, allow_pickle=False)
                np.save('linear_regression_coefs1k', linear_regression_coefs, allow_pickle=False)
                np.save('linear_regression_intercepts1k', linear_regression_intercepts, allow_pickle=False)
            # figz.data = []

        if ((n_svrs % 25) == 0):
            print(['n_svrs:  ' + str(n_svrs)])


# Get histogram of token - token distances for clustering:
#
dist = DistanceMetric.get_metric('manhattan')
token_dists = dist.pairwise(maternal_fetal_feature_vectors[0:200,:])

# token_dists = distance_matrix(maternal_fetal_feature_vectors, maternal_fetal_feature_vectors, p=1, threshold=100000000)
# token_dists = distance.cdist(maternal_fetal_feature_vectors[0:5,:], maternal_fetal_feature_vectors[0:5,:], metric='cityblock')
token_dists = distance.pdist(maternal_fetal_feature_vectors, metric='cityblock')

# token_dist_hist = np.histogram(token_dists, bins=1000)
# token_dist_hist_idxs = np.arange(len(token_dist_hist))

token_dists_sorted = np.sort(token_dists)
token_dist_idxs = np.arange(len(token_dists_sorted))
fig = make_subplots(rows=1, cols=1)
fig.append_trace(go.Scatter(x=token_dist_idxs, y=token_dists_sorted), row=1, col=1)
fig.show()

# kmeans_maternal_fetal = KMeans(n_clusters = 100, init = 'k-means++').fit(maternal_fetal_feature_vectors)

post_init = init_delay + n_coef_tpls - svr_wdw_lth   # Post-init processing w/ no gap

for svr_wdw_beg in np.arange(post_init, post_init + rem_record_lth - svr_wdw_lth, wdw_shift):

    wdw_beg = svr_wdw_beg
    wdw_end = wdw_beg + svr_wdw_lth
    fetal_lead_wdw = np.float32(np.zeros([(wdw_end - wdw_beg), ]))
    mat_lead_wdw = np.float32(np.zeros([(wdw_end - wdw_beg), ]))
    cwt_wdw = np.float32(np.zeros([(wdw_end - wdw_beg), n_feats]))
    cwt_wdw_fetal = np.float32(np.zeros([(wdw_end - wdw_beg), n_feats]))
    regr_idx = 0

    # Snapshot of maternal and fetal CWT contexts (templates) and CWT feature vectors for regression
    for wdw_idx in np.arange(wdw_beg, wdw_end):
        fetal_lead_wdw[regr_idx] = fetal_lead[wdw_idx]  # Extract lead windows for regression
        mat_lead_wdw[regr_idx] = mat_lead[wdw_idx]

        cwt_wdw[regr_idx, :] = cwt_trans[wdw_idx - cwt_wdw_lth_h: wdw_idx + cwt_wdw_lth_h -1,:].flatten()  # Extract feature vectors for regression & knn
        cwt_wdw_fetal[regr_idx, :] = cwt_trans[wdw_idx - cwt_wdw_lth_h: wdw_idx + cwt_wdw_lth_h -1,:].flatten()  # Extract feature vectors for regression & knn

        regr_idx = regr_idx + 1

    # Maternal & fetal CWT templates centered on this sample:
    maternal_feature_vector_s = cwt_wdw.flatten()
    maternal_feature_vector_rs = np.reshape(maternal_feature_vector_s, (1, maternal_feature_vector_s.size))

    maternal_fetal_feature_vector_s = np.concatenate((cwt_wdw.flatten(), cwt_wdw_fetal.flatten()), axis=None)

    # Get k-nn maternal lead templates:
    token_dists_knn = distance.cdist(maternal_feature_vector_rs, maternal_feature_vectors, metric='cityblock')
    token_dists_knn_fl = token_dists_knn.flatten()

    token_dists_knn_sorted_idxs = np.argsort(token_dists_knn).flatten()
    token_dists_knn_fl = token_dists_knn.flatten()
    token_dists_knn_sorted = token_dists_knn_fl[token_dists_knn_sorted_idxs]
    token_dist_knn_idxs = np.arange(len(token_dists_knn_sorted))

    #
    # fig = make_subplots(rows=1, cols=1)
    # fig.append_trace(go.Scatter(x=token_dist_knn_idxs, y=token_dists_knn_sorted), row=1, col=1)
    # fig.show()

    # Retrieve regression coef's from best matches:
    #
    nusv_lin_coef = linear_regression_coefs[token_dists_knn_sorted_idxs[0], :]
    nusv_intercept = linear_regression_intercepts[token_dists_knn_sorted_idxs[0]]
    
    # Generate abdominal signal estimates:
    #
    cwt_wdw_trans = np.transpose(cwt_wdw)
    z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
    z_cwt_xcoef_rs = z_cwt_xcoef + nusv_intercept

    # Update abdominal signal estimate:
    abdominal_est[overlap_wdw_idx : (overlap_wdw_idx + svr_wdw_lth)] = np.add(z_cwt_xcoef_rs, abdominal_est[overlap_wdw_idx: (overlap_wdw_idx + svr_wdw_lth)])

    # x_idxs = np.arange(len(fetal_lead_wdw))
    # figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal", "Abdominal Estimate"))
    # figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead_wdw), row=1, col=1)
    # figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead_wdw), row=2, col=1)
    # figz.append_trace(go.Scatter(x=x_idxs, y=z_cwt_xcoef_rs), row=3, col=1)
    # figz.show()
    # time.sleep(5.0)


    # maternal_feature_vectors[n_svrs, :] = cwt_wdw.flatten()
    # maternal_fetal_feature_vectors[n_svrs, :] = np.concatenate((cwt_wdw.flatten(), cwt_wdw_fetal.flatten()), axis=None)


    # nusv_res = NuSVR(nu=0.75, C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001,
    #                  cache_size=200, verbose=False, max_iter=-1)
    # z_rbf = nusv_res.fit(cwt_wdw, fetal_lead_wdw).predict(cwt_wdw)
    #
    # nusv_lin_coef = np.float32(nusv_res.coef_)
    # cwt_wdw_trans = np.transpose(cwt_wdw)
    # z_cwt_xcoef = np.matmul(nusv_lin_coef, cwt_wdw_trans)
    # z_cwt_xcoef_rs = np.reshape(z_cwt_xcoef, (svr_wdw_lth,)) + np.float32(nusv_res.intercept_)
    #
    # linear_regression_coefs[n_svrs, :] = np.float32(nusv_lin_coef)
    # linear_regression_intercepts[n_svrs] = np.float32(nusv_res.intercept_)


    if ((n_svrs % 50) == 1214):
        figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal",
                                                             "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4, Training Record Length = 5000",
                                                             "Abdominal Estimate"))
        # x_idxs = np.arange(len(fetal_lead))
        figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead_wdw), row=1, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead_wdw), row=2, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=z_cwt_xcoef_rs), row=3, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=z_rbf), row=3, col=1)
        figz.show()
        time.sleep(5.0)

    if ((n_svrs % 250) == 0):
        np.save('abdominal_est1k', abdominal_est, allow_pickle=False)

        x_idxs = np.arange(overlap_wdw_idx)
        figz = make_subplots(rows=3, cols=1, subplot_titles=("Maternal", "Abdominal", "Abdominal Estimate"))
        figz.append_trace(go.Scatter(x=x_idxs, y=mat_lead[0 : overlap_wdw_idx]), row=1, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=fetal_lead[0 : overlap_wdw_idx]), row=2, col=1)
        figz.append_trace(go.Scatter(x=x_idxs, y=abdominal_est[0 : overlap_wdw_idx]), row=3, col=1)
        figz.show()
        time.sleep(5.0)

    if ((n_svrs % 25) == 0):
        print(['n_svrs:  ' + str(n_svrs)])

    overlap_wdw_idx = overlap_wdw_idx + 1
    n_svrs = n_svrs + 1

arf = 12
# figz = make_subplots(rows=2, cols=1, subplot_titles=("Maternal", "Maternal NuSVR Estimate: nu=0.75, Linear, C=1.0, CWT Window Length = 4, Training Record Length = 5000"))
# figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
# figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=2, col=1)
# figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=2, col=1)
# figz.show()

# matplotlib.pyplot.close()

# Run trained SVR on full record:
#
wdw_beg = 1
wdw_end = 15000
regr_idx = 0
fetal_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
mat_lead_wdw = np.zeros([(wdw_end - wdw_beg),])
cwt_wdw = np.zeros([(wdw_end - wdw_beg), n_feats])
for wdw_idx in np.arange(wdw_beg, wdw_end):
    fetal_lead_wdw[regr_idx] = fetal_lead[wdw_idx]
    mat_lead_wdw[regr_idx] = mat_lead[wdw_idx]
    blef = cwt_trans[wdw_idx - cwt_wdw_lth_h : wdw_idx + cwt_wdw_lth_h -1, :]
    cwt_wdw[regr_idx,:] = blef.flatten()
    regr_idx = regr_idx +1

z_rbf = nusv_res.predict(cwt_wdw)
figz = make_subplots(rows=2, cols=1)
figz.append_trace(go.Scatter(x = x_idxs, y = mat_lead_wdw), row=1, col=1)
figz.append_trace(go.Scatter(x = x_idxs, y = fetal_lead_wdw), row=2, col=1)
figz.append_trace(go.Scatter(x = x_idxs, y = z_rbf), row=2, col=1)
figz.show()


# plt.plot(fetal_lead[500:700])
# plt.plot(svr_rbf.predict(cwt_trans[500:700,:]))

arf = 12



