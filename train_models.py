#!/usr/bin/env python
# coding: utf-8

# # Stage 1: Setup
# 
# - import packages and define util functions

# In[4]:


# %run inflation.py


# In[26]:


#from inflation import *
from Inflation_Forecast import *

# # Stage 2: Hyperparameter tuning
# 
# - 매월($m$) 첫째주 금요일을 기준으로 전월 예측변수($X_{m-1}$)를 이용하여 $h$-month forward 인플레이션율 인플레이션율($y_m(h)$)을 예측
# - 당월 인플레이션 전망은 $y_m(0)$, 3개월후 인플레이션 전망시 타겟변수는 $y_m(3)$
# - $y_m(h) = P_{cpi1}.shift(-h)$, $X_m = predictors.shift(1)$
# - 예측변수 결측치는 변수별 ffill, ext/rf/ar 등으로 보간
# - 월중 당월 데이터가 공표되는 변수는 당월 값을 예측변수로 이용 $X_m = predictors$하여 예측변수에 포함

# # Stage 3: 표본외 예측력 평가(매년 재추정)
# 
# - 2016년 1월부터 모형추정 및 실시간 전망 수행

# In[27]:


import time
import sys
sys.setrecursionlimit(10**9)

import glob
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
has3key=False

# In[28]:


#import s3fs

#bidas_fs = s3fs.S3FileSystem(anon=False)
#s3_repo_path = 's3://newtech/public/inf_nowcasting'
s3_repo_path = '/content/Inflation_Forecast'

#vintage_s3_dir = os.path.join(s3_repo_path, 'input/mdata')
#input_s3_dir = os.path.join(s3_repo_path, 'input')
#model_s3_dir = os.path.join(s3_repo_path, 'model')
#output_s3_dir = os.path.join(s3_repo_path, 'output')

vintage_s3_dir = '/content/Inflation_Forecast/input/mdata'
input_s3_dir = '/content/Inflation_Forecast/input'
model_dir = '/content/Inflation_Forecast/model'
output_dir = '/content/Inflation_Forecast/output'

saved_path = s3_repo_path
if not has_s3key():
    saved_path = set_local_path()
    #bidas_fs = s3fs.S3FileSystem(anon=True)


# In[4]:


today = datetime.now().strftime('%Y-%m-%d')

#vintages = pd.date_range('2012-12-02', today, freq='W-FRI')
#targets = pd.date_range('2016-01-31', '2023-12-31', freq='M')
#2018년 데이터만 
vintages = pd.date_range('2018-01-01', '2018-12-31', freq='W-FRI')
targets = pd.date_range('2018-01-01', '2018-12-31', freq='M')


# In[5]:


DF_vintages = {}

for date in vintages:
    df = pd.read_csv(f'{vintage_s3_dir}/{date:%Y-%m-%d}.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    DF_vintages[date] = df
    
# 과거 인플레이션 실제치
_, act = get_train_data_v5(DF_vintages[vintages[-1]])


# In[6]:


# global ens_pred, lm_pred, ext_pred, rw_pred, arima_pred


# In[7]:


# for MDA, 2018년만
tm16 = '2018-1-31'
#tm19 = '2019-1-31'
#tm20 = '2020-1-31'
#tm21 = '2021-1-31'
#tm22 = '2022-1-31'
tm99 = '2018-12-31'
#tm99 = '2023-10-31'


# In[ ]:


DO_TRAIN = False
DO_HYPERPARAMETER_TUNE = False # if False, only train best model
# GET_EVALUATION_PLOT = True

# model_types = ['reg', 'ext'] #['arima','rw', 'reg', 'ext']
# train_models(model_types, DO_HYPERPARAMETER_TUNE) 
# predict_and_plot(model_types, True)


# In[8]:


def train_models(model_types=['arima', 'rw', 'reg', 'ext'], is_tune=False):
    print("called train_models")
    DO_TRAIN = True
    DO_HYPERPARAMETER_TUNE = is_tune
    
    for model_type in model_types:
        print(">> Train Model : ", model_type)
        if model_type == 'arima':
            train_arima()
        elif model_type == 'rw':
            train_rw()
        elif model_type == 'reg':
            train_reg()
        elif model_type == 'ext':
            train_ext()
    
    # make ensemble models
    if 'reg' in model_types and 'ext' in model_types:
        predict_and_plot(['reg', 'ext']) # for ens model
                


# In[9]:


def predict_and_plot(model_types=['arima', 'rw', 'reg', 'ext'], get_plot=False):
    
    for model_type in model_types:
        print(">> Evaluate Model : ", model_type)
        if model_type == 'arima':
            arima_pred, arima_mae, arima_rmse = predict_arima()
        elif model_type == 'rw':
            rw_pred, rw_mae, rw_rmse = predict_rw()
        elif model_type == 'reg':
            lm_pred, lm_mae, lm_rmse = predict_reg()
        elif model_type == 'ext':
            ext_pred, ext_mae, ext_rmse = predict_ext()
    
    # make ensemble models
    if 'reg' in model_types and 'ext' in model_types:
        ens_pred, ens_mae, ens_rmse, ens_models = make_ensemble_models(lm_pred, lm_mae, lm_rmse, ext_pred, ext_mae, ext_rmse)
        
        if get_plot:
            print(">> Plot MDA")
            #plot_mda(ens_pred) # ens_pred쪽 수정하기!!

    # get mae_rmse plots
    if get_plot:
        print(">> Plot MAE/RMSE")
        if 'arima' in model_types and 'rw' in model_types:
            for _h in [0, 3, 12]:
                plot_mae_rmse(arima_mae, arima_rmse, rw_mae, rw_rmse, h=_h, best=5, good=10, title='ARIMA',
                              loc=1, ncol=4, figsize=(25, 8), fontsize=15, bbox_to_anchor=(1, -0.05))
        if 'reg' in model_types and 'rw' in model_types:
            for _h in [0, 3, 12]:
                plot_mae_rmse(lm_mae, lm_rmse, rw_mae, rw_rmse, h=_h, best=5, good=10, title='linear',
                              loc=1, ncol=2, figsize=(25, 6), fontsize=13, bbox_to_anchor=(1, -0.05))
        if 'ext' in model_types and 'rw' in model_types:
            for _h in [0, 3, 12]:
                plot_mae_rmse(ext_mae, ext_rmse, rw_mae, rw_rmse, h=_h, best=5, good=10, title='EXT (0 0 [6, 8, 10] 1)',
                              loc=1, ncol=3, figsize=(25, 6), fontsize=14, bbox_to_anchor=(1, -0.05))
        if 'reg' in model_types and 'ext' in model_types and 'rw' in model_types:
            for _h in [0, 3, 12]:
                plot_mae_rmse(ens_mae, ens_rmse, rw_mae, rw_rmse, h=0, best=5, good=10, title='Ensemble',
                              loc=1, ncol=6, figsize=(25, 8), fontsize=15, bbox_to_anchor=(1, -0.05))
                


# ## ARIMA

# In[10]:


def train_arima():
    param_combi = itertools.product([2, 3], [1, 2], [2, 3])
    if not DO_HYPERPARAMETER_TUNE:
        param_combi = itertools.product([3], [1], [3])
    
    start_time = time.time()
    
    for p, q, r in tqdm(param_combi):
        print('\np, q, r: ', [p, q, r])

        p_results = pd.DataFrame(index=vintages)

        for hor in [0, 3, 12]: # forecasting horizon
            print('\nh = ', hor, end=' => ')

            for tm in targets:
                if tm.month == 1:
                    print(tm.strftime('%Y-%m'), end=' ')
                #else:
                #    print(tm.strftime('%m'), end=' ')
                elapsed = time.time() - start_time
                mins = elapsed // 60
                secs = elapsed - (mins*60)
                #print(f'{tm:%Y-%m} ({int(mins):d}m{int(secs):d}s) for hor', end=' ')

                if hor == 0:
                    m0 = tm - DateOffset(months = 1) #타겟월 전월
                    m1 = tm - DateOffset(months = hor) + MonthEnd(0) #hor=0이면, m1=타겟월
                else:
                    m0 = tm - DateOffset(months = hor + 3) + MonthEnd(0)
                    m1 = tm - DateOffset(months = hor) + MonthEnd(0)

                tmp = vintages[Nth_friday(m0.year, m0.month, 1) <= vintages]
                dates = tmp[tmp <= Nth_friday(m1.year, m1.month, -1)]

                res = pd.DataFrame(index=dates, columns=[tm])
                for date in dates:

                    #X_train, y_train = get_train_data(date, hor, lag, dn, m1, rolling=rolling)
                    _, y = get_train_data_v5(DF_vintages[date])

                    #if tm.month in [1]:        # 타겟월이 1월인 경우만 모형 추정
                    #if tm.month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:  # 타겟월이 매분기 첫째월인 경우만 모형 추정
                    #y_train = y_train - y_train.shift(hor + 1)
                    y = y.shift(-hor)
                    arima = sm.tsa.arima.ARIMA(y, order=(p, q, r))
                    predicted = arima.fit().predict(end = tm).iloc[-1]
                    res.loc[date] = predicted

                res = pd.concat([res], axis=1, keys=[hor])
                p_results = pd.concat([p_results, res], axis=1)

        p_results.columns = pd.MultiIndex.from_tuples(p_results.columns)
        p_results = p_results.sort_index(axis=1)

        #fn = f'p_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 1년에 한번 모형추정
        #fn = f'p3_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 1분기에 한번 모형추정
        #fn = f'p12_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 매월 모형추정
        fn = f'arima_p{p}_q{q}_r{r}.pkl'  # 매월 모형추정


        fp = os.path.join(output_dir, fn)
        with open(fp, 'wb') as f:
            pickle.dump(p_results, f)


# ## RW

# In[11]:


def train_rw():
    start_time = time.time()

    p_results = pd.DataFrame(index=vintages)

    for hor in [0, 3, 12]: # forecasting horizon
        print('\nh = ', hor, end=' => ')

        for tm in targets:
            if tm.month == 1:
                print(tm.strftime('%Y-%m'), end=' ')
            #else:
            #    print(tm.strftime('%m'), end=' ')
            elapsed = time.time() - start_time
            mins = elapsed // 60
            secs = elapsed - (mins*60)
            #print(f'{tm:%Y-%m} ({int(mins):d}m{int(secs):d}s) for hor', end=' ')

            if hor == 0:
                m0 = tm - DateOffset(months = 1) #타겟월 전월
                m1 = tm - DateOffset(months = hor) + MonthEnd(0) #hor=0이면, m1=타겟월
            else:
                m0 = tm - DateOffset(months = hor + 3) + MonthEnd(0)
                m1 = tm - DateOffset(months = hor) + MonthEnd(0)

            tmp = vintages[Nth_friday(m0.year, m0.month, 1) <= vintages]
            dates = tmp[tmp <= Nth_friday(m1.year, m1.month, -1)]

            res = pd.DataFrame(index=dates, columns=[tm])
            for date in dates:

                _, y = get_train_data_v5(DF_vintages[date])
                y = y.shift(-hor)
                res.loc[date] = y.dropna().iloc[-1]

            res = pd.concat([res], axis=1, keys=[hor])
            p_results = pd.concat([p_results, res], axis=1)

    p_results.columns = pd.MultiIndex.from_tuples(p_results.columns)
    p_results = p_results.sort_index(axis=1)

    #fn = f'p_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 1년에 한번 모형추정
    #fn = f'p3_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 1분기에 한번 모형추정
    #fn = f'p12_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 매월 모형추정
    fn = f'rw.pkl'  # 매월 모형추정
    

        #fp = os.path.join(output_s3_dir, fn)
    fp = '/content/Inflation_Forecast/output_s3'
        #with bidas_fs.open(fp, 'wb') as f:
    with open(fp, 'wb') as f:
        pickle.dump(p_results, f)



# ## Reg

# In[12]:


def train_reg():
    lm_var0 = ['P_cpi_1', 'P_eir', 'GB_cp_3']
    lm_var0_0 = ['P_cpi_1', 'GB_cp_3']
    lm_var1 = ['P_ipi_1', 'P_ipi_2', 'P_ipi_6', 'L_eap_10', 'C_rsi_1',
               'C_css_1', 'IE_mir_9', 'RE_atpi_1', 'F_fb_1', 'P_adm', 'P_ppi_1']

    lm_models = []
    for c in [1, 2]:
        lm_models += [list(i) + lm_var0 for i in itertools.combinations(lm_var1, c)]
    
    lm_best_models = [[['P_ipi_6', 'RE_atpi_1'] + lm_var0, 4, 0, 0], # for lm_roll0_lag4_model, h=0
                  [['F_fb_1', 'P_adm'] + lm_var0, 4, 0, 0], # for lm_roll0_lag4_model, h=0
                  [['RE_atpi_1', 'P_adm'] + lm_var0, 6, 0, 0], # for lm_roll0_lag6_model, h=0
                  [['P_ipi_6', 'RE_atpi_1'] + lm_var0, 6, 0, 0], # for lm_roll0_lag6_model, h=0
                  [['RE_atpi_1'] + lm_var0, 3, 0, 3], # for lm_roll0_lag3_model, h=3
                  [['P_ipi_6', 'RE_atpi_1'] + lm_var0, 6, 0, 3], # for lm_roll0_lag6_model, h=3
                  [['P_ipi_6', 'RE_atpi_1'] + lm_var0_0, 4, 0, 12], # for lm_roll0_lag4_model, h=12
                  [['L_eap_10', 'P_adm'] + lm_var0, 6, 0, 12]] # for lm_roll0_lag6_model, h=12
    
    start_time = time.time()

    for model, lag, rolling in tqdm(itertools.product(lm_models, [0, 3, 4, 6], [0])):
        hors = [0, 3, 12]
        if not DO_HYPERPARAMETER_TUNE:
            flag = False
            for best_model in lm_best_models:
                
                _model, _lag, _rolling, _hor = best_model
                
                if set(model) == set(best_model[0]) and _lag == lag and _rolling == rolling:
                    hors = [_hor]
                    flag = True
                    
            if not flag:
                continue
        
        print('\nmodel, lag, rolling: ', model, lag, rolling)

        model_name = ':'.join(model)
        p_results = pd.DataFrame(index=vintages)

        for hor in hors: # forecasting horizon
            print('\nh:', hor, end = ' => ')

            for tm in targets:
                if tm.month == 1:
                    print(tm.strftime('%Y-%m'), end=' ')
                #else:
                #    print(tm.strftime('%m'), end=' ')
                elapsed = time.time() - start_time
                mins = elapsed // 60
                secs = elapsed - (mins*60)
                #print(f'{tm:%Y-%m} ({int(mins):d}m{int(secs):d}s) for hor', end=' ')

                if hor == 0:
                    m0 = tm - DateOffset(months = 1) #타겟월 전월
                    m1 = tm - DateOffset(months = hor) + MonthEnd(0) #hor=0이면, m1=타겟월
                else:
                    m0 = tm - DateOffset(months = hor + 3) + MonthEnd(0)
                    m1 = tm - DateOffset(months = hor) + MonthEnd(0)

                tmp = vintages[Nth_friday(m0.year, m0.month, 1) <= vintages]
                dates = tmp[tmp <= Nth_friday(m1.year, m1.month, -1)]

                res = pd.DataFrame(index=dates, columns=[tm])
                for date in dates:
                    X_train, y_train = get_train_data_v5(DF_vintages[date], lag, 3, False, m1, rolling=rolling, predictors=model)
                    eqn = 'y ~ 1 + ' + ' + '.join(X_train.columns.tolist())
                    #if tm.month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:  # 타겟월이 매분기 첫째월인 경우만 모형 추정
                    if tm.month in [1]:        # 타겟월이 1월인 경우만 모형 추정
                        y_train = y_train.shift(-hor).to_frame('y')
                        Xy_train = pd.concat([X_train, y_train], axis=1).dropna(axis=0)

                        reg = smf.ols(eqn, data = Xy_train).fit()
                        joblib.dump(reg, f"{model_dir}/lm_roll{rolling}_lag{lag}_model{model_name}_{tm.year}-01_h{hor}.pkl")
                        upload_to_s3(f"{model_dir}/lm_roll{rolling}_lag{lag}_model{model_name}_{tm.year}-01_h{hor}.pkl", model_s3_dir)
                        
                    predictor = X_train.loc[[m1]]
                    predicted = reg.predict(predictor)[0]
                    #predicted = predicted + y.loc[m1 - MonthEnd()]
                    res.loc[date] = predicted

                res = pd.concat([res], axis=1, keys=[hor])
                p_results = pd.concat([p_results, res], axis=1)

            #print('\n')
            #for i in reg.params.index:
            #    print(f"{i}: {reg.params.loc[i]:.2f}", end='   ')

        p_results.columns = pd.MultiIndex.from_tuples(p_results.columns)
        p_results = p_results.sort_index(axis=1)

        #fn = f'p_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 1년에 한번 모형추정
        #fn = f'p3_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 1분기에 한번 모형추정
        fn = f'lm_roll{rolling}_lag{lag}_model{model_name}.pkl'  # 매월 모형추정
        
        if has_s3key():
            fp = os.path.join(output_s3_dir, fn)
            with bidas_fs.open(fp, 'wb') as f:
                pickle.dump(p_results, f)
        else:
            fp = os.path.join(output_dir, fn)
            with open(fp, 'wb') as f:
                pickle.dump(p_results, f)


# ## EXT

# In[13]:


def train_ext():

    start_time = time.time()
    _hors = [0, 3, 12]
    
    combi = [[0, 0, 6, 1], [0, 0, 8, 1], [0, 0, 10, 1], [0, 0, 8, 2], [0, 0, 10, 2]]
    all_combi = [c + _hors for c in combi]
    
    if not DO_HYPERPARAMETER_TUNE:
        all_combi = [[0, 0, 8, 2, [0, 12]], [0, 0, 10, 1, [0, 3]], [0, 0, 6, 1, [12]]]

    for rolling, lag, md, dn, hors in tqdm(all_combi):
        print('\nrolling, lag, md, dn: ', [rolling, lag, md, dn])
        model = EXT(n_estimators=2000, max_features=1.0, max_depth=md, n_jobs=-1) # random_state=0

        p_results = pd.DataFrame(index=vintages)

        for hor in hors: # forecasting horizon
            print('\nh = ', hor, end = ' => ')

            for tm in targets:
                if tm.month == 1:
                    print(tm.strftime('%Y-%m'), end=' ')
                #else:
                #    print(tm.strftime('%m'), end=' ')
                elapsed = time.time() - start_time
                mins = elapsed // 60
                secs = elapsed - (mins*60)
                #print(f'{tm:%Y-%m} ({int(mins):d}m{int(secs):d}s) for hor', end=' ')

                if hor == 0:
                    m0 = tm - DateOffset(months = 1) #타겟월 전월
                    m1 = tm - DateOffset(months = hor) + MonthEnd(0) #hor=0이면, m1=타겟월
                else:
                    m0 = tm - DateOffset(months = hor + 3) + MonthEnd(0)
                    m1 = tm - DateOffset(months = hor) + MonthEnd(0)

                tmp = vintages[Nth_friday(m0.year, m0.month, 1) <= vintages]
                dates = tmp[tmp <= Nth_friday(m1.year, m1.month, -1)]

                res = pd.DataFrame(index=dates, columns=[tm])
                
                for date in dates:
                    X_train, y_train = get_train_data_v5(DF_vintages[date], lag, dn, True, m1, rolling=rolling)
                    if hor == 12:
                        X_train = X_train.drop(['P_eir'], axis = 1) #

                    if tm.month in [1]:  # 타겟월이 매분기 첫째월인 경우만 모형 추정
                        y_train = y_train.shift(-hor) - y_train.shift(1)
                        Xy_train = pd.concat([X_train, y_train], axis=1).dropna(axis=0)
                        model.fit(Xy_train.iloc[:-1, :-1], Xy_train.iloc[:-1, -1])
                        joblib.dump(model, f"{model_dir}/ext_roll{rolling}_lag{lag}_d{md}_g{dn}_{tm.year}-01_h{hor}.pkl")
                        upload_to_s3(f"{model_dir}/ext_roll{rolling}_lag{lag}_d{md}_g{dn}_{tm.year}-01_h{hor}.pkl", model_s3_dir)                                     

                    #_, y = get_train_data(vintages[-1], 0, 0, 0)
                    _, y = get_train_data_v5(DF_vintages[vintages[-1]])

                    predictor = X_train.loc[[m1]]
                    res.loc[date] = model.predict(predictor) + y.loc[m1 - MonthEnd()]

                res = pd.concat([res], axis=1, keys=[hor])
                p_results = pd.concat([p_results, res], axis=1)

        p_results.columns = pd.MultiIndex.from_tuples(p_results.columns)
        p_results = p_results.sort_index(axis=1)

        #fn = f'p_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 1년에 한번 모형추정
        #fn = f'p3_r{rolling}_l{lag}_d{md}_dn{dn}_rl_test.pkl'  # 1분기에 한번 모형추정
        fn = f'ext_roll{rolling}_lag{lag}_d{md}_g{dn}.pkl'  # p1_diff : 1월에만 cpi 차이 예측
        
        if has_s3key():
            fp = os.path.join(output_s3_dir, fn)
            with bidas_fs.open(fp, 'wb') as f:
                pickle.dump(p_results, f)
        else:
            fp = os.path.join(output_dir, fn)
            with open(fp, 'wb') as f:
                pickle.dump(p_results, f)


# # 모형별 예측치 읽어들이고, mae/rmse, last_pred 계산

# ## RW

# In[14]:


def predict_rw():
    rw_pred = pd.read_pickle(f'{output_s3_dir}/rw.pkl')
    
    if DO_TRAIN and not has_s3key(): 
        rw_pred = pd.read_pickle(f'{output_dir}/rw.pkl')

    rw_mae = pd.DataFrame()
    rw_rmse = pd.DataFrame()

    err, rw_mae, rw_rmse = align_error_by_week(get_error_by_vintage(rw_pred, act))
    rw_pred_last = get_pred_last(rw_pred)
    
    return rw_pred, rw_mae, rw_rmse


# ## ARIMA

# In[15]:


def predict_arima():
    files = bidas_fs.glob(f'{output_s3_dir}/arima_*')
    files = ["s3://"+file for file in files]
        
    if DO_TRAIN and not has_s3key():    
        files = glob.glob(f'{output_dir}/arima_*')
    
    names = [file.split('/')[-1].split('.')[0] for file in files]

    arima_pred = get_pred(files, names)
    arima_mae, arima_rmse, arima_pred_last = get_eval_last_pred(arima_pred, names, act)

    return arima_pred, arima_mae, arima_rmse


# ## Reg

# In[16]:


def predict_reg():
    lm_var0 = ['P_cpi_1', 'P_eir', 'GB_cp_3']
    lm_var0_0 = ['P_cpi_1', 'GB_cp_3']
    
    if DO_HYPERPARAMETER_TUNE:
        lm_models = [['L_eap_10'] + lm_var0,
         ['RE_atpi_1'] + lm_var0,
         ['F_fb_1', 'P_adm'] + lm_var0,
         ['L_eap_10', 'RE_atpi_1'] + lm_var0,
         ['RE_atpi_1'] + lm_var0,
         ['F_fb_1', 'P_adm'] + lm_var0,
         ['L_eap_10', 'P_adm'] + lm_var0,
         ['P_ipi_6', 'RE_atpi_1'] + lm_var0,
         ['RE_atpi_1', 'P_adm'] + lm_var0,
         ['P_ipi_6', 'RE_atpi_1'] + lm_var0,
         ['P_ipi_6', 'L_eap_10'] + lm_var0,
         ['IE_mir_9', 'RE_atpi_1'] + lm_var0,
         ['P_pip_6', 'RE_atpi_1'] + lm_var0_0]

        lm_pred_files = [f'lm_roll0_lag{l}_model{m}.pkl' for l in [0, 3, 4, 6] for m in [':'.join(model) for model in lm_models]]
    else:
        lm_models = [['P_ipi_6', 'RE_atpi_1'] + lm_var0, # for lm_roll0_lag4_model, h=0
          ['F_fb_1', 'P_adm'] + lm_var0, # for lm_roll0_lag4_model, h=0
          ['RE_atpi_1', 'P_adm'] + lm_var0, # for lm_roll0_lag6_model, h=0
          ['P_ipi_6', 'RE_atpi_1'] + lm_var0, # for lm_roll0_lag6_model, h=0
          ['RE_atpi_1'] + lm_var0, # for lm_roll0_lag3_model, h=3
          ['P_ipi_6', 'RE_atpi_1'] + lm_var0, # for lm_roll0_lag6_model, h=3
          ['P_ipi_6', 'RE_atpi_1'] + lm_var0_0, # for lm_roll0_lag4_model, h=12
          ['L_eap_10', 'P_adm'] + lm_var0] # for lm_roll0_lag6_model, h=12
    
        lm_pred_files = [f'lm_roll0_lag{l}_model{m}.pkl' for m, l in zip([':'.join(model) for model in lm_models], [4, 4, 6, 6, 3, 6, 6, 6])]
    
    files = bidas_fs.glob(f'{output_s3_dir}/lm_*')
    files = ["s3://"+file for file in files]
        
    if DO_TRAIN and not has_s3key():    
        files = glob.glob(f'{output_dir}/lm_*')
            
    files = [file for file in files if file.split('/')[-1] in lm_pred_files]
    names = [file.split('/')[-1].split('.')[0] for file in files]
    
    lm_pred = get_pred(files, names)
    lm_mae, lm_rmse, lm_pred_last = get_eval_last_pred(lm_pred, names, act)

    for df, data in zip([lm_pred, lm_mae, lm_rmse, lm_pred_last], ['pred', 'mae', 'rmse', 'pred_last']):
        df.to_pickle(f"{output_dir}/lm_{data}.pkl")
        upload_to_s3(f"{output_dir}/lm_{data}.pkl", output_s3_dir)
        dfs = []

    for data in ['pred', 'mae', 'rmse', 'pred_last']:
        df = pd.read_pickle(f"{output_dir}/lm_{data}.pkl")
        dfs.append(df)

    lm_pred = dfs[0]
    lm_mae = dfs[1]
    lm_rmse = dfs[2]
    lm_pred_last = dfs[3]
    
    return lm_pred, lm_mae, lm_rmse


# ## EXT

# In[17]:


def predict_ext():
    files = bidas_fs.glob(f'{output_s3_dir}/ext_*')
    files = ["s3://"+file for file in files]
        
    if DO_TRAIN and not has_s3key():    
        files = glob.glob(f'{output_dir}/ext_*')
        
    names = [file.split('/')[-1].split('.')[0] for file in files]
    
    ext_pred = get_pred(files, names)
    ext_mae, ext_rmse, ext_pred_last = get_eval_last_pred(ext_pred, names, act)
    
    return ext_pred, ext_mae, ext_rmse


# ## Ensemble

# In[18]:


def make_ensemble_models(lm_pred, lm_mae, lm_rmse, ext_pred, ext_mae, ext_rmse):
    ## LM
    lm_best_for_ensemble = []
    hors = lm_mae.columns.get_level_values(0).unique()

    for hor in hors:
        lm_best_for_ensemble += lm_mae[hor].iloc[-1].sort_values().index[:3].tolist()
        lm_best_for_ensemble += lm_rmse[hor].iloc[-1].sort_values().index[:3].tolist()

    lm_best_for_ensemble = list(set(lm_best_for_ensemble))
    #len(lm_best_for_ensemble) = 10
    
    ## ext
    ext_best_for_ensemble = []
    hors = ext_mae.columns.get_level_values(0).unique()

    for hor in hors:
        ext_best_for_ensemble += ext_mae[hor].iloc[-1].sort_values().index[:3].tolist()
        ext_best_for_ensemble += ext_rmse[hor].iloc[-1].sort_values().index[:3].tolist()

    ext_best_for_ensemble = list(set(ext_best_for_ensemble))
    
    ## ens
    ens_models = []
    for c in [1, 2, 3]:
        ens_models += [list(i) for i in itertools.combinations(lm_best_for_ensemble, c)]

    ens_models = [m + [n] for m in ens_models for n in ext_best_for_ensemble] + ens_models
    ens_models = sorted(ens_models)
    #len(ens_models) = 875
    
    hors = lm_pred.columns.get_level_values(1).unique().tolist()
    target = lm_pred.columns.get_level_values(2).unique().tolist()

    big_pred = pd.concat([lm_pred, ext_pred], axis=1)
    ens_pred = pd.DataFrame()

    for i, model in tqdm(enumerate(ens_models)):
        pred = big_pred[model]
        df = pd.DataFrame(0, index=vintages, columns=pd.MultiIndex.from_product([hors, targets]))
        for m in pred.columns.get_level_values(0).unique():
            df = df + pred[m]
        df = df / len(model)

        df = pd.concat([df], axis=1, keys=['ens' + str(i)])
        ens_pred = pd.concat([ens_pred, df], axis=1)

    ens_pred.to_pickle(f"{output_dir}/ens_pred.pkl")
    upload_to_s3(f"{output_dir}/ens_pred.pkl", output_s3_dir)
    
    ens_pred = ens_pred.sort_index(axis=1)
    ens_mae, ens_rmse, ens_pred_last = get_eval_last_pred(ens_pred, ens_pred.columns.get_level_values(0).unique(), act)
    # ens_pred.info()  566 X 325500, 1.4GB

    for df, data in zip([ens_pred, ens_mae, ens_rmse, ens_pred_last], ['pred', 'mae', 'rmse', 'pred_last']):
        df.to_pickle(f"{output_dir}/ens_{data}.pkl")
        upload_to_s3(f"{output_dir}/ens_{data}.pkl", output_s3_dir)
    
    dfs = []

    for data in ['pred', 'mae', 'rmse', 'pred_last']:
        df = pd.read_pickle(f"{output_dir}/ens_{data}.pkl")
        dfs.append(df)

    ens_pred = dfs[0]
    ens_mae = dfs[1]
    ens_rmse = dfs[2]
    ens_pred_last = dfs[3]
    
    return ens_pred, ens_mae, ens_rmse, ens_models


# In[1]:


def get_ens_models():
    #lm_files = bidas_fs.glob(f'{output_s3_dir}/lm_roll*')
    #ext_files = bidas_fs.glob(f'{output_s3_dir}/ext_roll*')
    lm_files = glob.glob('/content/Inflation_Forecast/output/lm_roll*')
    ext_files = glob.glob('/content/Inflation_Forecast/output/ext_roll*')
    if DO_TRAIN and not has_s3key():
        #lm_files = glob.glob(f'{output_dir}/lm_*')
        #ext_files = glob.glob(f'{output_dir}/ext_*')
        lm_files = glob.glob('/content/Inflation_Forecast/output/lm_*')
        ext_files = glob.glob('/content/Inflation_Forecast/output/ext_*')
    
    lm_files = [f.split('/')[-1].split('.')[0] for f in lm_files]
    ext_files = [f.split('/')[-1].split('.')[0] for f in ext_files]
    
    _ens_models = []
    for c in [1, 2, 3]:
        _ens_models += [list(i) for i in itertools.combinations(lm_files, c)]

    _ens_models = [m + [n] for m in _ens_models for n in ext_files] + _ens_models
        
    return _ens_models


# In[2]:


def get_best_ens_models(ens_models):
    ens_best = {0:0, 3:0, 12:0}    
    
    # h=0, 3, 12 일때 best 조합)
    for i, ens in enumerate(ens_models):
        if set(ens) == set(['lm_roll0_lag4_modelF_fb_1:P_adm:P_cpi_1:P_eir:GB_cp_3', 'lm_roll0_lag6_modelRE_atpi_1:P_adm:P_cpi_1:P_eir:GB_cp_3', 'ext_roll0_lag0_d10_g1']):
            # print("h0 == ", i, ens)
            ens_best[0] = i
        if set(ens) == set(['lm_roll0_lag3_modelRE_atpi_1:P_cpi_1:P_eir:GB_cp_3', 'lm_roll0_lag6_modelP_ipi_6:RE_atpi_1:P_cpi_1:P_eir:GB_cp_3', 'ext_roll0_lag0_d10_g1']):
            # print("h3 == ", i, ens)
            ens_best[3] = i
        #if set(ens) == set(['lm_roll0_lag4_modelP_ipi_6:RE_atpi_1:P_cpi_1:P_eir:GB_cp_3', 'ext_roll0_lag0_d6_g1']):
        if set(ens) == set(['lm_roll0_lag4_modelP_ipi_6:RE_atpi_1:P_cpi_1:GB_cp_3', 'ext_roll0_lag0_d6_g1']):
            # print("h12 == ", i, ens)
            ens_best[12] = i
            
    return ens_best


# # MDA

# ## Plot MDA

# In[21]:


### 수정필요
def plot_mda(ens_pred):
    MATCHED = {}

    M0 = tm16
    M1 = tm99
    
    best_ens = ['ens695', 'ens113', 'ens7']
    
    if not DO_HYPERPARAMETER_TUNE:
        best_ens = ['ens33', 'ens26', 'ens11']
    
    for h, model in zip([0, 3, 12], best_ens):
        mae, rmse, pred_last = get_eval_last_pred(ens_pred.loc[:, idx[[model], :, M0:M1]], [model], act)
        matched, mda, mda, p1, p2 = get_mda(pred_last[h][model], act, h, model=model, print_result=False)

        df = matched.to_frame('m')
        df['i'] = 0
        df.loc[df.m.eq(False), 'i'] = 1
        df['c'] = df['i'].cumsum()
        df = df.loc[df.m == True]
        df['date'] = df.index
        df = pd.concat([df.groupby('c')['date'].first().to_frame('first'), df.groupby('c')['date'].last().to_frame('last')], axis=1)
        df.index.names = [None]
        MATCHED[h] = df
        
    fig, axs = plt.subplots(3, 1, figsize=(20, 12))
    for ax, h in zip(axs, [0, 3, 12]):
        act[targets].loc['2017':].plot(ax=ax, lw=8, color='gray', marker='o', markersize=10, alpha=0.5)
        #ax.grid(which='minor', lw=1)
        ax.grid(which='major', lw=1)
        #ax.legend(fontsize=14)
        ax.set_title(f"h = {h}", fontsize=16)

        for ix, row in MATCHED[h].iterrows():
            m0 = row['first']
            m1 = row['last'] + MonthEnd(1)
            m0 = Nth_friday(m0.year, m0.month, -2)
            m1 = Nth_friday(m1.year, m1.month, 2)
            ax.fill_between(pd.date_range(m0, m1), 10, color='blue', alpha=.2)

    fig.tight_layout()


# In[ ]:




