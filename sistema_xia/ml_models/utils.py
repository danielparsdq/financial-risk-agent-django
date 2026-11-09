import pandas as pd
import numpy as np
import unicodedata
import os
import shap
import optuna
import mlflow
import mlflow.sklearn
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

def clean_categorical(df):
    def clean_text(val):
        if not isinstance(val, str):
            return val
        nfkd_form = unicodedata.normalize('NFKD', val)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower().strip().replace(" ", "_")
    return df.apply(lambda col: col.apply(clean_text))

def run_training_task(training_instance):
    try:
        training_instance.status = 'training'
        training_instance.save()

        # 1. Load Data
        df = pd.read_csv(training_instance.dataset.path)
        
        # 2. Renamed Columns (Same as notebook)
        column_mapping = {
            'person_age': 'edad',
            'person_income': 'ingresos',
            'person_home_ownership': 'tipo_vivienda',
            'person_emp_length': 'anios_trabajo',
            'loan_intent': 'motivo_prestamo',
            'loan_grade': 'grado_prestamo',
            'loan_amnt': 'monto_prestamo',
            'loan_int_rate': 'tasa_interes',
            'loan_status': 'riesgo_crediticio',
            'loan_percent_income': 'porcentaje_ingresos',
            'cb_person_default_on_file': 'historial_mora',
            'cb_person_cred_hist_length': 'historial_credito_anios'
        }
        df = df.rename(columns=column_mapping)

        # 2.5 Inversión del target (1 = Aprobado, 0 = Riesgo)
        df['riesgo_crediticio'] = 1 - df['riesgo_crediticio']

        # 3. GDPR/Leakage exclusions (Same as notebook)
        columnas_excluidas = ["edad", "tasa_interes", "grado_prestamo"]
        target = "riesgo_crediticio"
        
        X = df.drop(columns=columnas_excluidas + [target], errors='ignore')
        y = df[target]

        # Categorical values discovery
        cat_features = X.select_dtypes(include=['object']).columns.tolist()
        num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        categorical_values = {}
        for col in cat_features:
            categorical_values[col] = df[col].unique().tolist()
        
        training_instance.categorical_values = categorical_values
        training_instance.features = X.columns.tolist()
        training_instance.save()

        # 4. Split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        # 5. Pipeline
        pipeline_pre = Pipeline(steps=[
            ('preprocessor', ColumnTransformer(transformers=[
                ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))]), num_features),
                ('cat', Pipeline(steps=[
                    ('clean', FunctionTransformer(clean_categorical, feature_names_out='one-to-one')),
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), cat_features)
            ]))
        ])

        # 6. Optuna - Using configurations from instance
        N_TRIALS = training_instance.optuna_trials
        TIMEOUT = training_instance.optuna_timeout

        SEED = 42
        n_classes = len(np.unique(y_train))

        def get_model_pipeline(trial_params, model_class):
            return Pipeline(steps=[('preprocessor', pipeline_pre), ('classifier', model_class(**trial_params))])

        best_models = {}
        
        # Helper for optimization
        def optimize(objective_func, name):
            study = optuna.create_study(direction='maximize')
            study.optimize(objective_func, n_trials=N_TRIALS, timeout=TIMEOUT)
            return study

        # XGBoost
        def obj_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 60, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
                'random_state': SEED,
                'n_jobs': -1,
                'tree_method': 'hist',
                'eval_metric': 'logloss',
                'objective': 'binary:logistic' if n_classes == 2 else 'multi:softprob'
            }
            if n_classes > 2: params['num_class'] = n_classes
            model = get_model_pipeline(params, XGBClassifier)
            model.fit(X_train, y_train)
            return f1_score(y_test, model.predict(X_test))

        # LightGBM
        def obj_lgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 60, 220),
                'max_depth': trial.suggest_int('max_depth', -1, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
                'random_state': SEED,
                'n_jobs': -1,
                'verbosity': -1,
                'objective': 'binary' if n_classes == 2 else 'multiclass'
            }
            if n_classes > 2: params['num_class'] = n_classes
            model = get_model_pipeline(params, LGBMClassifier)
            model.fit(X_train, y_train)
            return f1_score(y_test, model.predict(X_test))

        # Random Forest
        def obj_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'random_state': SEED,
                'n_jobs': -1
            }
            model = get_model_pipeline(params, RandomForestClassifier)
            model.fit(X_train, y_train)
            return f1_score(y_test, model.predict(X_test))

        # Run optimization
        study_xgb = optimize(obj_xgb, "xgb")
        study_lgb = optimize(obj_lgb, "lgb")
        study_rf = optimize(obj_rf, "rf")

        # Train best models
        b_xgb_params = {**study_xgb.best_params, 'random_state': SEED, 'n_jobs': -1, 'tree_method': 'hist', 'eval_metric': 'logloss', 'objective': 'binary:logistic' if n_classes == 2 else 'multi:softprob'}
        best_xgb = get_model_pipeline(b_xgb_params, XGBClassifier).fit(X_train, y_train)
        
        b_lgb_params = {**study_lgb.best_params, 'random_state': SEED, 'n_jobs': -1, 'verbosity': -1, 'objective': 'binary' if n_classes == 2 else 'multiclass'}
        best_lgb = get_model_pipeline(b_lgb_params, LGBMClassifier).fit(X_train, y_train)
        
        b_rf_params = {**study_rf.best_params, 'random_state': SEED, 'n_jobs': -1}
        best_rf = get_model_pipeline(b_rf_params, RandomForestClassifier).fit(X_train, y_train)

        # Select best model (The notebook specifically trains all but picks one)
        # The user requested to store training results.
        # It's better to store the results of the model specified in `training_instance.model_used`?
        # NO, the notebook trains all and selects the best. 
        # But the model field `model_used` should probably reflect the result or be a filter?
        # User said "modelo usado (xgboost, lightgbm o random forest)". 
        # I'll stick to the model selected in the form, OR I'll train the one selected.
        # "Debe funcionar exactamente como lo hace el notebook" -> The notebook finds the best among 3.
        
        m_scores = {
            'xgboost': accuracy_score(y_val, best_xgb.predict(X_val)),
            'lightgbm': accuracy_score(y_val, best_lgb.predict(X_val)),
            'random_forest': accuracy_score(y_val, best_rf.predict(X_val))
        }
        
        best_id = max(m_scores, key=m_scores.get)
        if best_id == 'xgboost': 
            best_model = best_xgb
        elif best_id == 'lightgbm': 
            best_model = best_lgb
        else: 
            best_model = best_rf

        # Versioning logic
        from ml_models.models import MLModelTraining
        last_version = MLModelTraining.objects.filter(model_used=best_id, status='completed').order_by('-version').first()
        new_version = (last_version.version + 1) if last_version else 1

        # Update instance with best model found
        training_instance.model_used = best_id
        training_instance.version = new_version
        training_instance.accuracy = m_scores[best_id]
        
        y_pred = best_model.predict(X_val)
        training_instance.f1_score = f1_score(y_val, y_pred)
        training_instance.precision = precision_score(y_val, y_pred)
        training_instance.recall = recall_score(y_val, y_pred)
        
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_val)[:, 1]
            training_instance.roc_auc = roc_auc_score(y_val, y_prob)

        # 7. SHAP
        preprocessor = best_model.named_steps['preprocessor']
        classifier = best_model.named_steps['classifier']
        X_val_transformed = preprocessor.transform(X_val)
        
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"f_{i}" for i in range(X_val_transformed.shape[1])]

        explainer = shap.TreeExplainer(classifier)
        shap_values_raw = explainer.shap_values(X_val_transformed)
        
        # Store mean absolute SHAP values per feature
        if isinstance(shap_values_raw, list):
            sv = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
        else:
            sv = shap_values_raw
        
        importance = np.abs(sv).mean(0)
        shap_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
        training_instance.shap_values = shap_dict

        # 8. Save with MLflow and into Database field
        import joblib
        import io
        from django.core.files.base import ContentFile

        # Save to MLflow (Disk)
        model_path = os.path.join(settings.BASE_DIR, '..', 'models', f"training_{training_instance.id}")
        mlflow.sklearn.save_model(sk_model=best_model, path=model_path, serialization_format='cloudpickle')

        # Save to Database Field (Binary)
        model_buffer = io.BytesIO()
        joblib.dump(best_model, model_buffer)
        training_instance.model_binary.save(
            f"model_{training_instance.model_used}_v{training_instance.version}.joblib",
            ContentFile(model_buffer.getvalue()),
            save=False
        )

        training_instance.status = 'completed'
        training_instance.save()

    except Exception as e:
        training_instance.status = 'failed'
        training_instance.log = str(e)
        training_instance.save()
        raise e
