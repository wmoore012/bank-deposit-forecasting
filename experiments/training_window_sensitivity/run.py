"""Exploratory fixed-holdout training-window sensitivity; never overwrites core results."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/window-mpl')
import ast,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
FEATURES=['log_deposits','cash_ratio','loan_ratio','equity_ratio','prior_growth']
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()
# Reuse the exact existing data preparation and network routines.
needed={'add_adjacent_reports','add_model_fields','fit_preprocessor','transform_features','build_network','fit_network','predict_log_growth'}
source=(ROOT/'build_masterclass.py').read_text()
for literal in ast.walk(ast.parse(source)):
 if isinstance(literal,ast.Constant) and isinstance(literal.value,str) and 'def ' in literal.value:
  try: tree=ast.parse(literal.value)
  except SyntaxError: continue
  for node in tree.body:
   if isinstance(node,ast.FunctionDef) and node.name in needed:
    routine=ast.unparse(node)
    if node.name=='fit_network':
     # Keep scalar targets as (batch, 1), including a final batch of one.
     routine=routine.replace('(X_train, y_train)', '(X_train, y_train.reshape(-1, 1))').replace('(X_valid, y_valid)', '(X_valid, y_valid.reshape(-1, 1))')
    exec(compile(routine,'<existing experiment routine>','exec'))
    needed.remove(node.name)
assert not needed,needed
raw=pd.read_csv(ROOT/'data/fdic_financials_2013_2024.csv')
reporting=pd.read_csv(ROOT/'data/fdic_reporting_2010_2024.csv')
p=add_adjacent_reports(raw.merge(reporting,on=['CERT','REPDTE'],how='left',validate='one_to_one'))
keep=p.BKCLASS.isin(['N','NM','SM','SB','SI','SL']) & p.CALLFORM.isin([31,41,51]) & (p.quarter_number-p.prior_quarter).eq(1) & (p.next_quarter-p.quarter_number).eq(1) & p.prior_deposits.gt(0) & p.DEPDOM.gt(0) & p.next_deposits.gt(0) & p.ASSET.gt(0)
rows=add_model_fields(p.loc[keep].copy())
valid=rows.loc[rows.date.between('2023-01-01','2023-09-30')].copy()
test=rows.loc[rows.date.between('2024-01-01','2024-09-30')].copy()
assert len(valid)==13834 and len(test)==13532
base=pd.read_csv(ROOT/'growth_outputs/masterclass/predictions.csv')
assert list(zip(test.CERT,test.date.dt.strftime('%Y-%m-%d')))==list(zip(base.CERT,base.date))
assert np.allclose(test.growth,base.growth)
plan={'status':'Exploratory: 2024 already inspected; no fresh holdout','training_starts':[2013,2020,2021],'training_end':'2022-09-30','validation_predictors':'2023 Q1-Q3','test_predictors':'2024 Q1-Q3','seeds':[42,7,99],'rationale':'2020 and 2021 test more recent training, with unchanged 2023 selection and 2024 scoring. No 2023-2026 observations moved into training. One bank does not establish an economy-wide regime.','source_sha256':hashlib.sha256((ROOT/'data/fdic_financials_2013_2024.csv').read_bytes()).hexdigest(),'features':FEATURES,'architecture':[5,32,16,1],'epochs_max':200,'patience':10,'batch_size':512,'learning_rate':0.001,'ridge_alphas':[.01,.1,1,10,100]}
(OUT/'plan.json').write_text(json.dumps(plan,indent=2))
scores=[];predictions=[]
def measure(pred):
 err=100*(pred-test.growth.to_numpy())
 audit=test[['CERT','date','growth']].copy();audit['pred']=pred
 precision=[]
 for _,q in audit.groupby('date'):
  k=int(np.ceil(len(q)*.1));actual=set(q.sort_values(['growth','CERT']).head(k).CERT);chosen=set(q.sort_values(['pred','CERT']).head(k).CERT);precision.append(len(actual&chosen)/k)
 return {'MAE_pp':float(np.abs(err).mean()),'RMSE_pp':float(np.sqrt(np.mean(err**2))),'bottom_decile_precision':float(np.mean(precision))}
for year in plan['training_starts']:
 train=rows.loc[rows.date.between(f'{year}-01-01','2022-09-30')].copy()
 if year==2013: assert len(train)==214425
 assert train.target_date.max()<valid.date.min() and valid.target_date.max()<test.date.min()
 imputer,scaler=fit_preprocessor(train)
 X_train=transform_features(train,imputer,scaler);X_valid=transform_features(valid,imputer,scaler);X_test=transform_features(test,imputer,scaler)
 y_train=train.log_growth.to_numpy(dtype='float32');y_valid=valid.log_growth.to_numpy(dtype='float32')
 candidates=[]
 for alpha in plan['ridge_alphas']:
  model=Ridge(alpha=alpha).fit(X_train,y_train)
  val=float(np.abs(100*(np.expm1(y_valid)-np.expm1(model.predict(X_valid)))).mean())
  candidates.append((val,alpha,model))
 _,alpha,ridge=min(candidates,key=lambda v:(v[0],v[1]))
 pred=np.expm1(ridge.predict(X_test));scores.append({'training_start':year,'model':'Ridge','seed':None,'training_rows':len(train),'alpha':alpha,**measure(pred)})
 for seed in plan['seeds']:
  mlp,history,elapsed=fit_network(seed);pred=np.expm1(predict_log_growth(mlp,X_test));metrics=measure(pred)
  scores.append({'training_start':year,'model':'MLP','seed':seed,'training_rows':len(train),'best_epoch':int(np.argmin(history['val_loss'])+1),'epochs':len(history['loss']),'seconds':elapsed,**metrics})
  saved=test[['CERT','date','growth']].copy();saved['prediction']=pred;saved['training_start']=year;saved['seed']=seed;predictions.append(saved)
  print(year,seed,metrics,flush=True)
 pd.DataFrame(scores).to_csv(OUT/'scores.csv',index=False)
zero=measure(np.zeros(len(test)));zero['bottom_decile_precision']=None
(OUT/'zero_baseline.json').write_text(json.dumps(zero,indent=2))
pd.concat(predictions).to_csv(OUT/'predictions.csv',index=False)
original=[s for s in scores if s['training_start']==2013 and s['model']=='MLP' and s['seed']==42][0]
assert abs(original['MAE_pp']-3.627317978)<.0001,original
print('Original MLP reproduced; zero baseline:',zero,flush=True)
