import pandas as pd
import tensorflow as tf
import xgboost as xgb
import numpy as np
import mlflow
import mlflow.keras
import mlflow.xgboost
import sys

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt


class MonteCarloDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

class AFMM:
    def __init__(self,params):
        self.params = params
        
    def data_prep(self,split_name,build_norm=False):
        norm_cols = []
        X_df = self.params['data'][split_name]['X_df'].copy()
        X_df.fillna(0,inplace=True)
        for col in X_df.columns:
            if col in self.params['parameters']['TF']['emb_layers']:
                if X_df[col].dtype == object:
                    X_df.loc[:,col] = X_df[col].astype('category')
                    #X_df.loc[:,col] = X_df.loc[:,col].cat.add_categories('NoVal')
                else:
                    X_df.loc[:,col] = X_df[col].astype('int32')
            else:
                if X_df[col].dtype == int:
                    X_df.loc[:,col] = X_df[col].astype('int32')
                else:
                    X_df.loc[:,col] = X_df[col].astype('float32')
                if build_norm == True:
                    if (X_df[col].max() > 10) | (X_df[col].min() < -10):
                        norm_cols += [col]
        
        if build_norm == True:
            self.params['models']['norm'] = {'model':StandardScaler().fit(X_df[norm_cols]),
                                             'norm_cols':norm_cols}
        else:
            norm_cols = self.params['models']['norm']['norm_cols']
        
        X_df.loc[:,norm_cols] = self.params['models']['norm']['model'].transform(X_df[norm_cols])
        self.params['data'][split_name]['X_df'] = X_df
        return self.params

    def XG_data(self,split_name,balance=False):
        if balance == True:
            cw = class_weight.compute_sample_weight(class_weight='balanced',y=self.params['data'][split_name]['y_df'])
        else:
            cw = None
            
        self.params['data'][split_name]['XG'] = xgb.DMatrix(self.params['data'][split_name]['X_df'],
                                                            label = self.params['data'][split_name]['y_df'],
                                                            feature_names = self.params['data'][split_name]['X_df'].columns,
                                                            weight = cw, enable_categorical = True)
        return self.params
        
    def XG_train(self,params, test=True):

        xgb_params = {
            'learning_rate': params['learning_rate'],
            'min_child_weight': params['min_child'],
            #'rate_drop': params['rate_drop'],
            'eta': params['eta'],
            'colsample_bytree': params['colsample_bytree'],
            'max_depth': int(params['max_depth']),
            'subsample': params['subsample'],
            'lambda': params['lambda'],
            'gamma': params['gamma'],
            'booster' : params['booster'],
            'eval_metric':params['eval_metric']
        }
        
        XG_mod = xgb.train(xgb_params, self.params['data']['train']['XG'], 1000, self.params['data']['validate']['watchlist'],
                            early_stopping_rounds=5, maximize=False, verbose_eval=0)
        
        yhat = XG_mod.predict(self.params['data']['validate']['XG'])
        
        if self.params['parameters']['objective'] == 'regression':
            loss = mean_squared_error(self.params['data']['validate']['y_df'],yhat)**.5
            print('RMSE:',loss)
        else:
            loss = 1 - f1_score(self.params['data']['validate']['y_df'],np.clip(np.round(yhat),a_min=0,a_max=1))
            print('F1-loss:',loss)
        
        if params['mlflow'] != False:
            with mlflow.start_run():
                mlflow.log_param('learning_rate',params['learning_rate'])
                mlflow.log_param('min_child_weight',params['min_child'])
                #mlflow.log_param('rate_drop',params['rate_drop'])
                mlflow.log_param('eta',params['eta'])
                mlflow.log_param('colsample_bytree',params['colsample_bytree'])
                mlflow.log_param('max_depth',int(params['max_depth']))
                mlflow.log_param('subsample',params['subsample'])
                mlflow.log_param('lambda',params['lambda'])
                mlflow.log_param('gamma',params['gamma'])
                if self.params['parameters']['objective'] == 'regression':
                    mlflow.log_metric("RMSE",loss)
                    mlflow.log_metric("r2",r2_score(self.params['data']['validate']['y_df'],yhat))
                    #mlflow.log_metric("accuracy",accuracy_score(self.params['data']['pandas']['y_val'],np.round(yhat)))
                else:
                    mlflow.log_metric("F1-loss",loss)
                    mlflow.log_metric("binary_crossentropy",log_loss(self.params['data']['validate']['y_df'],np.clip(yhat,a_min=0,a_max=1),eps = 1e-7))
                    mlflow.log_metric("roc_auc_score",roc_auc_score(self.params['data']['validate']['y_df'],np.clip(np.round(yhat),a_min=0,a_max=1)))
                    mlflow.log_metric("accuracy",accuracy_score(self.params['data']['validate']['y_df'],np.clip(np.round(yhat),a_min=0,a_max=1)))
                if test != True:
                    mlflow.xgboost.log_model(XG_mod,"models")
                
        if test == True:
            sys.stdout.flush()
            return {'loss': loss, 'status': STATUS_OK}
        else:
            features = XG_mod.get_score(importance_type='gain')
            feature_df = pd.DataFrame(
                data=list(features.values()),
                index=list(features.keys()),
                columns=['score']).sort_values(by='score', ascending=False)
            
            feature_df.nlargest(10, columns='score').plot(kind='barh', figsize=(16,8))
            return XG_mod, feature_df
    
    def XG_trial(self):
        hyperparams = {
            'learning_rate': hp.uniform('learning_rate', .001,.3),
            'max_depth': hp.uniform('max_depth', 3,30),
            #'rate_drop': hp.uniform('rate_drop', 0,.5),
            'min_child': hp.uniform('min_child', 0,1),
            'subsample': hp.uniform('subsample', 0,1),
            'colsample_bytree': hp.uniform('colsample_bytree', .5,1),
            'eta': hp.uniform('eta', 0,.5),
            'gamma': hp.uniform('gamma', 0,.3),
            'lambda': 1.,
            'booster': 'gbtree',
            'mlflow': False
        }
        
        if params['parameters']['objective'] == 'regression':
            hyperparams['eval_metric'] = 'rmse'
        else:
            hyperparams['eval_metric'] = 'logloss'
            
        for param in hyperparams.keys():
            if param not in self.params['parameters']['XG'].keys():
                self.params['parameters']['XG'][param] = hyperparams[param]
        
        if self.params['parameters']['objective'] == "classification":
            self.params = self.XG_data('train',balance=True)
        else:
            self.params = self.XG_data('train')
        self.params = self.XG_data('validate')
        self.params = self.XG_data('test')
        self.params['data']['validate']['watchlist'] = [(self.params['data']['train']['XG'], 'train'),
                                                          (self.params['data']['validate']['XG'], 'val')]
        
        if self.params['parameters']['XG']['mlflow'] != False:
            mlflow.set_experiment(self.params['parameters']['XG']['mlflow'])
        
        trials = Trials()
        best = fmin(self.XG_train, self.params['parameters']['XG'], algo=tpe.suggest, trials=trials,
                    max_evals=self.params['parameters']['XG']['trials'])
        
        for param in best.keys():
            self.params['parameters']['XG'][param] = best[param]
        if "misc" not in self.params['data']:
            self.params['data']['misc'] = {}
        self.params['models']['XG_model'], self.params['data']['misc']['feature_df'] = self.XG_train(self.params['parameters']['XG'],test=False)
        return self.params
    
    def TF_dataset(self,split_name,feature_mods,loss_functions=False):
        split = self.params['data'][split_name]
        split['X_df']['XG_yhat'] = self.params['models']['XG_model'].predict(split['XG'])
        split['X_df']['XG_yhat'] = split['X_df']['XG_yhat'].astype('float32')
        
        tfds = split['X_df'][feature_mods].copy()
        tfds.columns = tfds.columns  + "_out"
        
        tfds[self.params['parameters']['TF']['target'] + '_aux_out'] =  split['y_df']
        tfds[self.params['parameters']['TF']['target'] + '_out'] =  split['y_df']
        
        if loss_functions == True:
            loss_functions = {}
            for col in tfds:
                if (isinstance(tfds[col].iloc[0],np.int32)) & (max(tfds[col]) <= 1):
                    loss_functions[col] = 'binary_crossentropy'
                else:
                    loss_functions[col] = 'mse'
            if self.params['parameters']['objective'] == "classification":
                loss_functions["XG_yhat_out"] = 'binary_crossentropy'
                loss_functions[self.params['parameters']['TF']['target'] + "_aux_out"] = 'binary_crossentropy'
                loss_functions[self.params['parameters']['TF']['target'] + "_out"] = 'binary_crossentropy'
            else:
                loss_functions["XG_yhat_out"] = 'mse'
                loss_functions[self.params['parameters']['TF']['target'] + "_aux_out"] = 'mse'
                loss_functions[self.params['parameters']['TF']['target'] + "_out"] = 'mse'
            self.params['parameters']['TF']['loss_functions'] = loss_functions
                
        self.params['data'][split_name]['TF'] = tf.data.Dataset.from_tensor_slices((dict(split['X_df']),dict(tfds)))
        return self.params
        
    def TF_compile_data(self,params,build_datasets=[]):
        if type(params['feature_models']) == dict:
            feature_mods = list(params['feature_models'].keys())
            params['feature_layers'] = params['feature_models']
        else:
            if type(params['emb_layers']) == dict:
                emb_layers = list(params['emb_layers'].keys())
            else:
                emb_layers = params['emb_layers']
                
            if (type(params['feature_models']) == int) | (type(params['feature_models']) == float):
                feature_mods = list(self.params['data']['misc']['feature_df']
                                    .drop(emb_layers)
                                    .index[:int(params['feature_models'])])
                feature_mods += ['XG_yhat']
            else:
                feature_mods = params['feature_models']
                
            params['feature_layers'] = {col:[col]+emb_layers
                                        for col in feature_mods}
            params['feature_layers'][params['target'] + "_aux"] = emb_layers
        
        for dataset in build_datasets:
            if dataset == 'train':
                self.TF_dataset(dataset,feature_mods,loss_functions=True)
                params['loss_functions'] = self.params['parameters']['TF']['loss_functions']
            else:
                self.TF_dataset(dataset,feature_mods,loss_functions=False)
            
        return params
    
    def TF_build_layers(self,params):
        dropout_schedule = [0,0.03125,0.0625,0.125,0.1875,0.25,0.375,0.5]
        hidden_layers = []
        if params['dropout_shape'] == 'increasing':
            for i in range(int(params['layers'])):
                doi = dropout_schedule.index(params['dropout'])+i
                if doi > 7: doi = 7
                hidden_layers.append([params['neurons'],dropout_schedule[doi]])
            hidden_layers.append([params['fin_layer'],dropout_schedule[doi]])

        elif params['dropout_shape'] == 'decreasing':
            for i in range(int(params['layers'])):
                doi = dropout_schedule.index(params['dropout'])-i
                if doi < 0: doi = 0
                hidden_layers.append([params['neurons'],dropout_schedule[doi]])
            doi -= 1
            if doi < 0: doi = 0
            hidden_layers.append([params['fin_layer'],dropout_schedule[doi]])

        else:
            for i in range(int(params['layers'])):
                hidden_layers.append([params['neurons'],params['dropout']])
            hidden_layers.append([params['fin_layer'],params['dropout']])
        return hidden_layers

    def TF_add_model(self,params,feature_outputs=None,model_name=None,activation='linear'):
        if params['batch_norm'] == True:
            layer = layers.BatchNormalization()(feature_outputs)
        else:
            layer = feature_outputs

        for i in range(len(params['hidden_layers'])):
            
            if (model_name == self.params['parameters']['TF']['target']) & (i == 0):
                layer = layers.Dense(params['hidden_layers'][i][0],activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l1(params['L1']),
                                     name=model_name+'_Layer'+str(i))(layer)
            else:
                layer = layers.Dense(params['hidden_layers'][i][0],activation='relu',
                                     name=model_name+'_Layer'+str(i))(layer)
            
            last_layer = layer
            
            if params['batch_norm'] == True:
                layer = layers.BatchNormalization()(layer)
            if params['hidden_layers'][i][1] > 0:
                #layer = layers.Dropout(params['hidden_layers'][i][1])(layer)
                layer = MonteCarloDropout(params['hidden_layers'][i][1])(layer)
                
        output_layer = layers.Dense(1, activation=activation,
                                    name=model_name+'_out')(layer)
        return last_layer, output_layer

    def TF_add_emb(self,model_inputs={}):
        emb_inputs = {}
        emb_features = []

        for key,value in self.params['parameters']['TF']['emb_layers'].items():
            emb_inputs[key] = model_inputs[key]
            catg_col = feature_column.categorical_column_with_vocabulary_list(key, value)
            dim = int(len(value)**0.25)
            emb_col = feature_column.embedding_column(
                catg_col,dimension=dim)
            emb_features.append(emb_col)

        emb_layer = layers.DenseFeatures(emb_features)
        emb_outputs = emb_layer(emb_inputs)

        return emb_outputs

    def TF_feature_models(self,params,feature_layers,loss_functions):
        model_inputs = {}
        all_features = list(self.params['data']['train']['X_df'].columns)
        
        for feature in all_features:
            if feature in [k for k,v in self.params['parameters']['TF']['emb_layers'].items()]:
                if self.params['data']['train']['X_df'][feature].dtypes.name == 'category':
                    model_inputs[feature] = tf.keras.Input(shape=(1,), name=feature,dtype='string')
                else:
                    model_inputs[feature] = tf.keras.Input(shape=(1,), name=feature,dtype='int32')
            else:
                model_inputs[feature] = tf.keras.Input(shape=(1,), name=feature)
                
        if len(self.params['parameters']['TF']['emb_layers']) > 0:
            emb_outputs = self.TF_add_emb(model_inputs)
            
        output_layers = []
        eng_layers = []
        
        for key,value in feature_layers.items():
            feature_columns = [feature_column.numeric_column(f) for f in all_features if f not in value]
            feature_layer = layers.DenseFeatures(feature_columns)
            feature_outputs = feature_layer({k:v for k,v in model_inputs.items() if k not in value})
            
            if len(self.params['parameters']['TF']['emb_layers']) > 0:
                feature_outputs = layers.concatenate([feature_outputs,emb_outputs])
                
            if loss_functions[key+"_out"] == 'binary_crossentropy':
                act = 'sigmoid'
            else:
                act = 'linear'
            last_layer, output_layer = self.TF_add_model(params,feature_outputs=feature_outputs,model_name=key,activation=act)
            
            output_layers.append(output_layer)
            eng_layers.append(last_layer)
            
            if key == "XG_yhat":
                eng_layers.append(output_layer)
            
            if key == self.params['parameters']['TF']['target'] + "_aux":
                eng_layers.append(feature_outputs)
                eng_layers.append(output_layer)
        
        return model_inputs, output_layers, eng_layers
    
    def TF_train(self,params,trial=True,plot_hist=False):
        
        params['hidden_layers'] = self.TF_build_layers(params)
        params = self.TF_compile_data(params,build_datasets=['train','validate','test'])
                        
        model_inputs, output_layers, eng_layers = self.TF_feature_models(params,params['feature_layers'],params['loss_functions'])
        concat_layer = layers.concatenate(eng_layers)
        
        if self.params['parameters']['objective'] == "classification":
            act = 'sigmoid'
        else:
            act = 'linear'
            
        last_layer, output_layer = self.TF_add_model(
            params,
            feature_outputs=concat_layer,
            model_name=self.params['parameters']['TF']['target'],
            activation=act)
        
        output_layers.append(output_layer)
        
        model = tf.keras.Model(
            inputs=[model_inputs],
            outputs=output_layers)
        
        batches = [32,64,128,256,512,1024,2048]
        #batches = [1024,2048]
        
        weight = 1
        decline = (1 - params['feature_wgts'])/(batches.index(params['batch_size'])+1)
        for batch in batches:
            if batch > params['batch_size']:
                break
            else:
                weight -= decline
                aux_loss_wgt = weight / (len(params['feature_layers']) - 2)
                loss_wgts = [aux_loss_wgt for i in range(len(params['feature_layers']) - 2)]
                loss_wgts.append((1-weight)*0.15)
                loss_wgts.append((1-weight)*0.15)
                loss_wgts.append((1-weight)*0.70)
                
                model.compile(loss=params['loss_functions'], optimizer=tf.keras.optimizers.Adam(
                    learning_rate=params['learning_rate']),loss_weights=loss_wgts,metrics=['accuracy'])
                
                model.fit(self.params['data']['train']['TF'].batch(batch),
                          validation_data=self.params['data']['validate']['TF'].batch(params['batch_size']),
                          epochs=1, verbose=0)
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              mode='min',
                                              verbose=1,
                                              patience=params['patience'],
                                              restore_best_weights=True)
        
        history = model.fit(self.params['data']['train']['TF'].batch(params['batch_size']),
                            validation_data=self.params['data']['validate']['TF'].batch(params['batch_size']),
                            epochs=params['epochs'],
                            verbose=0,
                            callbacks=[es])
        
        if trial == True:
            loss, loss_variance = self.TF_evaluation(params,model=model,trial=trial)
            sys.stdout.flush()
            print('trial complete')
            return {'loss': loss, 'loss_variance':loss_variance, 'status': STATUS_OK}
        else:
            loss = self.TF_evaluation(params,model=model,trial=trial,plot_hist=history)
            self.params['parameters']['TF'] = params
            self.params['models']['TF_model'] = model
            return self.params
        
    def TF_evaluation(self,params,model=None,trial=False,plot_hist=False):
        #yhat = model.predict(params['valset'].batch(params['batch_size']))[-1]
        y_probas = np.stack([model.predict(self.params['data']['validate']['TF'].batch(params['batch_size']))[-1] for sample in range(params['samples'])])
        yhat = y_probas.mean(axis=0)
        losses = y_probas.copy()
        
        if self.params['parameters']['objective'] == "classification":
            losses = np.stack([1 - f1_score(self.params['data']['validate']['y_df'],np.clip(np.round(yh),a_min=0,a_max=1)) for yh in losses])
            loss = losses.mean(axis=0)
            print('F1-loss:',loss)
        else:
            losses = np.stack([mean_squared_error(self.params['data']['validate']['y_df'],yhat)**0.5 for yh in losses])
            loss = losses.mean(axis=0)
            print('RMSE:',loss)
            
        variance = np.var(losses, ddof=1)
            
        if plot_hist != False:
            plt.plot(plot_hist.history["loss"])
            plt.plot(plot_hist.history["val_loss"])
            plt.xlabel("Epochs")
            plt.ylabel("loss")
            plt.legend(["loss", "val_loss"])
            plt.show()
            
        if trial==True:
            if self.params['parameters']['TF']['mlflow'] != False:
                with mlflow.start_run():
                    mlflow.log_param('layers',params['layers'])
                    mlflow.log_param('neurons',params['neurons'])
                    mlflow.log_param('fin_layer',params['fin_layer'])
                    mlflow.log_param('L1',params['L1'])
                    mlflow.log_param('dropout',params['dropout'])
                    mlflow.log_param('dropout_shape',params['dropout_shape'])
                    mlflow.log_param('batch_norm',params['batch_norm'])
                    mlflow.log_param('feature_wgts',params['feature_wgts'])
                    mlflow.log_param('feature_models',params['feature_models'])
                    mlflow.log_param('batch_size',params['batch_size'])
                    mlflow.log_param('learning_rate',params['learning_rate'])
                    if self.params['parameters']['objective'] == 'regression':
                        mlflow.log_metric("RMSE",loss)
                        mlflow.log_metric("r2",r2_score(self.params['data']['validate']['y_df'],yhat))
                        #mlflow.log_metric("accuracy",accuracy_score(self.params['data']['pandas']['y_val'],np.round(yhat)))
                    else:
                        mlflow.log_metric("F1-loss",loss)
                        mlflow.log_metric("binary_crossentropy",log_loss(self.params['data']['validate']['y_df'],np.clip(yhat,a_min=0,a_max=1),eps = 1e-7))
                        mlflow.log_metric("roc_auc_score",roc_auc_score(self.params['data']['validate']['y_df'],np.clip(np.round(yhat),a_min=0,a_max=1)))
                        mlflow.log_metric("accuracy",accuracy_score(self.params['data']['validate']['y_df'],np.clip(np.round(yhat),a_min=0,a_max=1)))
                    mlflow.log_metric('variance',variance)
                    mlflow.keras.log_model(model,"models")
            else:
                hyperopt_df = pd.read_csv(params['path'] + 'hyperopt_trials.csv',index_col=0)
                
                hyperopt_df = hyperopt_df.append(pd.DataFrame(
                    columns=['layers','neurons','fin_layer','L1','dropout','dropout_shape','batch_norm',
                             'feature_wgts','feature_models','batch_size','learning_rate','variance','loss'],
                    data=[[params['layers'],params['neurons'],params['fin_layer'],params['L1'],
                           params['dropout'],params['dropout_shape'],params['batch_norm'],params['feature_wgts'],
                           params['feature_models'],params['batch_size'],params['learning_rate'],variance,loss]]))
                hyperopt_df.to_csv(params['path'] + 'hyperopt_trials.csv')
                self.params['data']['misc']['TF_trials'] = hyperopt_df
                
                if len(hyperopt_df) == 1:
                    model.save('/tmp/AFMM.h5')
                    self.params['models']['TF_model'] = model
                else:
                    if loss == min(hyperopt_df['loss']):
                        print("new best model")
                        model.save('/tmp/AFMM.h5')
                        self.params['models']['TF_model'] = model
                        self.params['models']['TF_model'] = model
                        #self.params['data']['test']['TF'] = params['testset']
            return loss, variance
        
    def TF_trials(self):
        
        hyperparams = {
            'layers': hp.quniform('layers',1.5,6.5,1),
            'neurons': hp.choice('neurons',[128,256,512,1024]),
            'fin_layer': hp.choice('fin_layer',[16,32]),
            'dropout': hp.choice('dropout',[0,0.125,0.25,0.375,0.5]),
            'dropout_shape': hp.choice('dropout_shape',['decreasing','increasing','flat']),
            'feature_wgts': hp.uniform('feature_wgts',0.25,0.5),
            'feature_models': hp.quniform('feature_models',10,60,1),
            'L1': hp.uniform('L1',0.0,0.3),
            'batch_norm': hp.choice('batch_norm', [True,False]),
            'target': self.params['data']['train']['y_df'].columns[0],
            'emb_layers': [],
            'batch_size': 2048,
            'learning_rate':0.0001,
            'patience':20,
            'epochs':500,
            'samples':100,
            #'trial':True,
            'mlflow': False,
            'spark_trials': False
        }
        
        for param in hyperparams.keys():
            if param not in self.params['parameters']['TF'].keys():
                self.params['parameters']['TF'][param] = hyperparams[param]
        
        if len(self.params['parameters']['TF']['emb_layers']) > 0:
            self.params['parameters']['TF']['emb_layers'] = {emb: self.params['data']['train']['X_df'][emb].unique()
                                                             for emb in self.params['parameters']['TF']['emb_layers']}
            #print(self.params['parameters']['TF']['emb_layers'])
        
        if self.params['parameters']['TF']['mlflow'] != False:
            mlflow.set_experiment(self.params['parameters']['TF']['mlflow'])
        
        if self.params['parameters']['TF']['spark_trials'] != False:
            trials = SparkTrials(parallelism = self.params['parameters']['TF']['spark_trials'])
        else:
            trials = Trials()
        
        best = fmin(self.TF_train, self.params['parameters']['TF'], algo=tpe.suggest, max_evals=self.params['parameters']['TF']['trials'], trials=trials)
        
        for param in best.keys():
            self.params['parameters']['TF'][param] = best[param]
        return self.params
        
    def TF_retrain(self,model=None,build_datasets=[],recompile=False):
        params = self.params['parameters']['TF']
        
        if len(build_datasets) > 0:
            params = self.TF_compile_data(params,build_datasets=build_datasets)
            self.params['parameters']['TF'] = params
            
        if recompile == True:
            aux_loss_wgt = params['feature_wgts'] / (len(params['feature_layers']) - 2)
            loss_wgts = [aux_loss_wgt for i in range(len(params['feature_layers']) - 2)]
            loss_wgts.append((1-params['feature_wgts'])*0.15)
            loss_wgts.append((1-params['feature_wgts'])*0.15)
            loss_wgts.append((1-params['feature_wgts'])*0.7)

            model.compile(loss=params['loss_functions'], optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                          loss_weights=loss_wgts,metrics=['accuracy'])
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=params['patience'],restore_best_weights=True)
        history = model.fit(self.params['data']['train']['TF'].batch(params['batch_size']),
                            validation_data=self.params['data']['validate']['TF'].batch(params['batch_size']),
                            epochs=params['epochs'], verbose=0, callbacks=[es])
        
        self.TF_evaluation(params,model=model,trial=False,plot_hist=history)
        
        if len(build_datasets) > 0:
            return model, self.params
        else:
            return model
        
    def predict(self,split='predict',build_dataset=True):
        if build_dataset == True:
            self.params['parameters']['TF'] = self.TF_compile_data(self.params['parameters']['TF'],build_datasets=[split])
            
        self.params['data'][split]['y_probas'] = np.stack([self.params['models']['TF_model']
                                                           .predict(self.params['data'][split]['TF'].batch(self.params['parameters']['TF']['batch_size']))[-1]
                                                           for sample in range(self.params['parameters']['TF']['samples'])])
        
        self.params['data'][split]['variance'] = np.var(self.params['data'][split]['y_probas'], ddof=1)
        
        self.params['data'][split]['yhat'] = pd.DataFrame(index=self.params['data'][split]['X_df'].index,
                                                          columns=['yhat'],
                                                          data=self.params['data'][split]['y_probas'].mean(axis=0))
        
        return self.params
