from django.contrib import admin
from django.utils.html import format_html
from .models import MLModelTraining
from .utils import run_training_task
import json

@admin.register(MLModelTraining)
class MLModelTrainingAdmin(admin.ModelAdmin):
    list_display = ('model_used', 'version', 'accuracy', 'f1_score', 'status', 'training_date')
    list_filter = ('model_used', 'status')
    readonly_fields = ('model_used', 'version', 'accuracy', 'f1_score', 'roc_auc', 'precision', 'recall', 'shap_values', 'features', 'categorical_values', 'status', 'log', 'training_date', 'model_binary')
    actions = ['start_training']

    fieldsets = (
        ('Configuración Optuna', {
            'fields': ('optuna_trials', 'optuna_timeout')
        }),
        ('Info de Salida', {
            'fields': ('model_used', 'version', 'status', 'training_date', 'model_binary')
        }),
        ('Dataset', {
            'fields': ('dataset',)
        }),
        ('Métricas Resultantes', {
            'fields': ('accuracy', 'f1_score', 'roc_auc', 'precision', 'recall')
        }),
        ('Datos Detallados', {
            'fields': ('features', 'categorical_values', 'shap_values')
        }),
        ('Logs de Error', {
            'fields': ('log',)
        }),
    )

    def save_model(self, request, obj, form, change):
        # Primero guardamos el objeto para tener el archivo dataset disponible
        super().save_model(request, obj, form, change)
        
        # Si es un objeto nuevo (no una edición de uno existente)
        if not change:
            import threading
            # Ejecutamos el entrenamiento en un hilo separado para no bloquear la UI
            thread = threading.Thread(target=run_training_task, args=(obj,))
            thread.start()
            self.message_user(request, "🚀 Entrenamiento iniciado en segundo plano. El estado se actualizará automáticamente.")

    def start_training(self, request, queryset):
        for instance in queryset:
            if instance.status != 'training':
                try:
                    run_training_task(instance)
                    self.message_user(request, f"Training completed for {instance}")
                except Exception as e:
                    self.message_user(request, f"Training failed for {instance}: {str(e)}", level='error')
    
    start_training.short_description = "🚀 Ejecutar entrenamiento"

    def get_readonly_fields(self, request, obj=None):
        if obj: # editing
            return self.readonly_fields + ('dataset',)
        return self.readonly_fields

