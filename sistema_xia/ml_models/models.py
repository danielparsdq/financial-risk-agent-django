from django.db import models
import json

class MLModelTraining(models.Model):
    MODEL_CHOICES = [
        ('xgboost', 'XGBoost'),
        ('lightgbm', 'LightGBM'),
        ('random_forest', 'Random Forest'),
    ]

    model_used = models.CharField(max_length=50, choices=MODEL_CHOICES, null=True, blank=True, help_text="Se seleccionará automáticamente al finalizar")
    version = models.PositiveIntegerField(default=1)
    
    # Configuración de entrenamiento
    optuna_trials = models.PositiveIntegerField(default=20, help_text="Número de pruebas para la optimización")
    optuna_timeout = models.PositiveIntegerField(default=60, help_text="Tiempo máximo (segundos) por modelo")

    # Metrics
    accuracy = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    roc_auc = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    
    # Values
    shap_values = models.JSONField(null=True, blank=True)
    features = models.JSONField(null=True, blank=True)
    categorical_values = models.JSONField(null=True, blank=True)
    
    # Dataset and results
    dataset = models.FileField(upload_to='ml_datasets/')
    model_binary = models.FileField(upload_to='ml_models_binaries/', null=True, blank=True)
    training_date = models.DateTimeField(auto_now_add=True)
    
    status = models.CharField(max_length=20, default='pending') # pending, training, completed, failed
    log = models.TextField(blank=True)

    class Meta:
        ordering = ['-training_date']

    def __str__(self):
        return f"{self.model_used or 'Pendiente'} v{self.version} - {self.training_date}"

    def predict(self, input_data):
        """
        Realiza una predicción utilizando el modelo almacenado.
        input_data: dict o DataFrame con las columnas necesarias.
        """
        import joblib
        import pandas as pd
        import io
        import shap
        from sklearn.pipeline import Pipeline

        if not self.model_binary:
            raise ValueError("El modelo no ha sido entrenado o no tiene un binario guardado.")

        # Cargar el modelo desde el FileField
        model_file = self.model_binary.open('rb')
        model = joblib.load(model_file)
        model_file.close()

        # Asegurar que sea un DataFrame
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        else:
            df_input = input_data

        # Realizar predicción
        prediction = model.predict(df_input)
        
        # Calcular SHAP local si es posible
        local_shap = []
        try:
            # Intentar obtener el explainer
            # Para XGBoost y LightGBM suele funcionar TreeExplainer
            explainer = shap.TreeExplainer(model)
            # Si el modelo tiene un pipeline, hay que explicar el clasificador final con los datos transformados
            if hasattr(model, 'steps'):
                # Si es un Pipeline de sklearn
                classifier = model.steps[-1][1]
                preprocessor = Pipeline(model.steps[:-1])
                X_transformed = preprocessor.transform(df_input)
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_transformed)
                
                # Manejar multiclase o binario
                if isinstance(shap_values, list): # Multiclase
                    instance_shap = shap_values[int(prediction[0])][0]
                else: # Binario o array
                    instance_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                
                # Mapear nombres de columnas si están disponibles
                try:
                    feat_names = preprocessor.get_feature_names_out()
                except:
                    feat_names = [f"feature_{i}" for i in range(len(instance_shap))]
                
                for name, val in zip(feat_names, instance_shap):
                    local_shap.append({'feature': name, 'value': float(val)})
            else:
                # Caso directo (menos probable con este proyecto)
                shap_values = explainer.shap_values(df_input)
                instance_shap = shap_values[0]
                for name, val in zip(df_input.columns, instance_shap):
                    local_shap.append({'feature': name, 'value': float(val)})
        except Exception as e:
            # Si falla el cálculo SHAP en tiempo real, se usará el global como fallback
            pass

        # Opcional: Proporción de probabilidad
        try:
            probs = model.predict_proba(df_input)
            return {
                'prediction': int(prediction[0]),
                'probability': probs[0].tolist(),
                'label': 'RIESGO' if prediction[0] == 1 else 'APROBADO',
                'local_shap': local_shap
            }
        except:
            return {
                'prediction': int(prediction[0]),
                'label': 'RIESGO' if prediction[0] == 1 else 'APROBADO',
                'local_shap': local_shap
            }

