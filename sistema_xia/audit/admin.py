from django.contrib import admin
from django.urls import path
from .models import AgentTransactionLog
from .views import audit_dashboard, audit_data_api

@admin.register(AgentTransactionLog)
class AgentTransactionLogAdmin(admin.ModelAdmin):
	list_display = (
		"created_at", 
		"agent_name", 
		"status", 
		"ai_risk_level", 
		"model_version",
		"legal_basis", 
		"pii_detected",
		"token_usage"
	)
	list_filter = (
		"status", 
		"agent_name", 
		"ai_risk_level", 
		"legal_basis", 
		"pii_detected",
		"created_at"
	)
	search_fields = ("agent_name", "endpoint", "error_message", "model_version")
	ordering = ("-created_at",)
	readonly_fields = (
		"agent_name",
		"endpoint",
		"request_payload",
		"response_payload",
		"status",
		"error_message",
		"created_at",
		"ai_risk_level",
		"model_version",
		"human_oversight",
		"transparency_confirmed",
		"token_usage",
		"execution_time_ms",
		"legal_basis",
		"pii_detected",
		"data_minimization_applied",
		"anonymized_payload",
		"consent_record_id",
		"data_retention_days",
	)

	def get_urls(self):
		urls = super().get_urls()
		custom_urls = [
			path('dashboard/', self.admin_site.admin_view(audit_dashboard), name='audit-dashboard'),
			path('api/data/', self.admin_site.admin_view(audit_data_api), name='audit-api-data'),
		]
		return custom_urls + urls

	# Agregar botón al listado de registros
	change_list_template = "admin/audit/agenttransactionlog_changelist.html"

	def has_add_permission(self, request):
		return False

	def has_delete_permission(self, request, obj=None):
		"""Solo permite eliminar si el usuario es superusuario."""
		return request.user.is_superuser

	actions = ["clear_all_logs"]

	@admin.action(description="Vaciar todos los logs de transacciones")
	def clear_all_logs(self, request, queryset):
		"""Acción para vaciar todos los logs, restringida a superusuarios."""
		if not request.user.is_superuser:
			self.message_user(request, "No tienes permisos para realizar esta acción.", level="error")
			return
		
		count = AgentTransactionLog.objects.all().count()
		AgentTransactionLog.objects.all().delete()
		self.message_user(request, f"Se han eliminado exitosamente {count} logs.", level="success")

	def has_change_permission(self, request, obj=None):
		return False

	def has_delete_permission(self, request, obj=None):
		return False
