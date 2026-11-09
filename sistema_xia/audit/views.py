from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
from .models import AgentTransactionLog
from django.db.models import Count, Avg, Sum
import json
from django.core.serializers.json import DjangoJSONEncoder
from django.http import JsonResponse

@staff_member_required
def audit_dashboard(request):
    logs = AgentTransactionLog.objects.all().values(
        'agent_name', 'status', 'ai_risk_level', 'legal_basis', 
        'pii_detected', 'human_oversight', 'token_usage', 
        'execution_time_ms', 'created_at'
    )
    logs_json = json.dumps(list(logs), cls=DjangoJSONEncoder)
    legal_basis_labels_json = json.dumps(dict(AgentTransactionLog.LEGAL_BASIS_CHOICES), cls=DjangoJSONEncoder)
    context = {
        'title': 'Dashboard de Auditoría IA',
        'logs_json': logs_json,
        'legal_basis_labels_json': legal_basis_labels_json,
    }
    return render(request, 'admin/audit/dashboard.html', context)

@staff_member_required
def audit_data_api(request):
    logs = AgentTransactionLog.objects.all().values(
        'agent_name', 'status', 'ai_risk_level', 'legal_basis', 
        'pii_detected', 'human_oversight', 'token_usage', 
        'execution_time_ms', 'created_at'
    )
    return JsonResponse(list(logs), safe=False)
