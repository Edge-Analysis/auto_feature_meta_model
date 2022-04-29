# Auto Feature Meta Model

Feature engineering is the most important step for improving model accuracy.  This model uses neural networks to automatically engineer new features with the overall goal of improving accuracy.  These features are higher order and often better than a modeler could produce manually.
Please see this article for the general idea and theory behind this model: https://towardsdatascience.com/automated-feature-engineering-using-neural-networks-5310d6d4280a

### New features since the article was written
+ This model now trains an XGBoost model first to provide a meta-model prediction (result feeds into neural network). It also returns feature importance so the neural network can choose top features automatically rather than the modeler manually selecting layers (and exclusions). A manual option is still available.
+ data_prep(): This model will now automatically detect columns that should be normalized, trains the scaler on training data, and transforms each dataset.
+ XG_data(): This model now automatically creates the DMatrix transformations necessary to train an XGBoost model with watchlists. It also includes an option to automatically apply class weights for unbalanced classification problems.
+ XG_mod(): Now includes option for mlflow logging.
+ XG_trial(): Auto builds hyperopt experiment to find best model. You can optionally include your own paramaters or parameter spaces.
+ TF_dataset(): Model now automatically creates the tensorflow dataset, greatly reducing the difficulty of using this model. This function automatically creates auxiliary outputs based on XGBoost feature importance. It also has an option to automatically create loss functions for each auxiliary output.
+ TF_compile_data(): Creates the blueprint for each neural network feature model, then calls the TF_dataset() function.
+ TF_build_layers(): now has a dropout list rather than using random space. These numbers produce even and efficient layers when paired with neuron list described later.  This change allows for quicker hyperopt convergence and is more mathematically clean.
+ TF_add_model(): Now uses a custom Monte Carlo dropout layer.  This allows for the dropout layer to remain active at prediction time.  In practice, this transforms the neural network into an ensemble model of many different networks (final result is the mean of n predictions).  It also allows for model uncertainty to be determined by the variance in predictions.
+ TF_add_emb(): No changes.
+ TF_feature_models(): Now supports int and string categorical columns for embeddings. Creates a "skip connection" on target feature that feeds into the final model.  This allows the neural network to learn short and longer networks on the target feature and creates a slight boosting effect. It also feeds the XGBoost prediction to the final model, hence the "meta model" name.
+ TF_train(): Now has the ability to create tensorflow datasets when running hyperopt. This change was made because now the number of feature models is a tunable hyperparameter and any change in this feature requires different model inputs.  Based on recent findings on model training speed and performance, this function now has a batchsize ramp up built in (starting at 32 and doubling each epoch up to a max of 2048). Finally, it also dynamically adjusts the loss weights to focus on learning the auxiliary outputs first, and gradually shifting to focus on the target output.
+ TF_evaluation(): Now includes mlflow logging and has been adjusted to allow for Monte Carlo dropout predictions (see TF_add_model notes above). Also returns variance to assist with hyperopt convergence.
+ TF_trials(): Auto builds hyperopt experiment to find best model. You can optionally include your own parameters or parameter spaces.
+ TF_retrain(): Allows for simple retraining on pretrained or imported models.
+ predict(): simplifies generating a final prediction and variance for imported models.


### Inputs
This model now takes JSON inputs to direct the training process.
##### Minimal inputs example:

params = {
    'data':{
        'train':{'X_df':X_train,'y_df':y_train},
    },
    'parameters':{
        'XG' : {},
        'TF' : {'target':y_train.columns[0]},
        'objective':'classification'
    },
    'models' : {}    
}

##### Full inputs example:
params = {
    'data':{
        'train':{'X_df':X_train,'y_df':y_train,'XG':dtrain,'TF':trainset},
        'validate':{'X_df':X_val,'y_df':y_val,'XG':dval,'TF':valset},
        'test':{'X_df':X_test,'y_df':y_test,'XG':dtest,'TF':testset},
        'predict:{<same format>}
        'misc':{'feature_df':XG_top_feature_df}
    },
    'parameters':{
        'XG':{
            'max_depth' : hp.uniform('max_depth', 10,40),
            'eta' : hp.uniform('eta', 0,.2),
            'subsample' : hp.uniform('subsample', 0.5,1),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,0.9),
            'gamma' : hp.uniform('gamma', 0.1,0.4),
            'learning_rate' : hp.uniform('learning_rate', .0001,.05),
            'min_child' : hp.uniform('min_child', 0.1,1),
            'subsample' : hp.uniform('subsample', 0.5,1),
            'trials':100,
            'mlflow':<path>
        },
        'TF':{
            'batch_norm':True,
            'dropout': hp.choice('dropout',[0.25,0.375,0.5]),
            'dropout_shape': 'flat',
            'feature_wgts': 0.4,
            'fin_layer':16,
            'trials':30,
            'epochs':500,
            'emb_layers':['fee_type','state','cc_nbr','nics_4_cd'],
            'learning_rate':0.00005,
            'batch_size': 2048,
            'mlflow':<path>
        },
        'objective':'regression'
    },
    'models':{'XG_model':xgmod,'TF_model':tfmod,'norm':prefit_norm_model}
}

### Future improvements:
+ auto multicollinearity detection for excluding features in feature models
+ auto shapley value generation
+ write update article and walkthrough.
