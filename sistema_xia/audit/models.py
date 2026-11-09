from django.db import models


class AgentTransactionLog(models.Model):
	# Enums para reportes tabulables (Compliance)
	RISK_LEVEL_CHOICES = [
		('low', 'Bajo'),
		('medium', 'Medio'),
		('high', 'Alto'),
		('unacceptable', 'Inaceptable'),
	]

	LEGAL_BASIS_CHOICES = [
		('consent', 'Consentimiento'),
		('none_yet', 'Aún no definido'),
		('legal_obligation', 'Obligación Legal'),
		('public_interest', 'Interés Público/Tarea Administrativa'),
		('pre_contractual', 'Fase Precontractual'),
		('legitimate_interest', 'Interés Legítimo'),
		('contract', 'Ejecución Contractual'),
	]

	agent_name = models.CharField(max_length=120)
	endpoint = models.CharField(max_length=120)
	request_payload = models.JSONField()
	response_payload = models.JSONField(null=True, blank=True)
	status = models.CharField(max_length=20, default="success")
	error_message = models.TextField(blank=True)
	created_at = models.DateTimeField(auto_now_add=True)

	# --- XIA Europa (AI Act) Compliance Fields ---
	ai_risk_level = models.CharField(
		max_length=20, 
		choices=RISK_LEVEL_CHOICES, 
		default='low',
		help_text="Categoría de riesgo según la Ley de IA de la UE"
	)
	model_version = models.CharField(max_length=50, default="gemma-4", help_text="Identificador único y versión del modelo/agente")
	human_oversight = models.BooleanField(default=False, help_text="Indica si hubo supervisión humana (Human-in-the-loop)")
	transparency_confirmed = models.BooleanField(default=True, help_text="Garantiza que el usuario sabe que interactúa con una IA")
	token_usage = models.IntegerField(default=0, help_text="Métrica cuantitativa de consumo de recursos")
	execution_time_ms = models.IntegerField(default=0, help_text="Latencia de respuesta para auditoría de eficiencia")

	# --- LOPDP Ecuador Compliance Fields ---
	legal_basis = models.CharField(
		max_length=50, 
		choices=LEGAL_BASIS_CHOICES, 
		default='public_interest',
		help_text="Base legal para el tratamiento de datos según LOPDP"
	)
	pii_detected = models.BooleanField(default=False, help_text="Detección automática de Datos de Carácter Personal (PII)")
	data_minimization_applied = models.BooleanField(default=True, help_text="Cumplimiento del principio de minimización")
	anonymized_payload = models.BooleanField(default=True, help_text="Indica si los datos sensibles fueron anonimizados")
	consent_record_id = models.CharField(max_length=100, null=True, blank=True, help_text="Vínculo al registro de consentimiento")
	data_retention_days = models.IntegerField(default=365, help_text="Días restantes para la eliminación/bloqueo de datos")

	class Meta:
		ordering = ["-created_at"]
		verbose_name = "Registro de Transacción del Agente"
		verbose_name_plural = "Registros de Transacciones del Agente"

	def __str__(self) -> str:
		return f"{self.agent_name} [{self.status}] at {self.created_at:%Y-%m-%d %H:%M:%S}"
