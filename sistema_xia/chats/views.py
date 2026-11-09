from django.shortcuts import render
from django.contrib.admin.views.decorators import staff_member_required
from django.http import JsonResponse
from .models import Chats, Messages
from django.db.models import Count, Q
import json
from django.core.serializers.json import DjangoJSONEncoder

@staff_member_required
def chat_dashboard_api(request):
    # Obtener chats con conteo de mensajes y datos básicos
    chats = Chats.objects.all().annotate(
        msg_count=Count('messages')
    ).values(
        'id', 'user__username', 'title', 'created_at', 'msg_count'
    )
    
    # Obtener distribución de roles de mensajes
    msg_roles = Messages.objects.values('role').annotate(count=Count('id'))
    
    # Últimos mensajes para ver actividad reciente
    recent_activity = Messages.objects.order_by('-created_at')[:50].values(
        'chat_id', 'role', 'created_at', 'chat__title'
    )

    data = {
        'chats': list(chats),
        'msg_roles': list(msg_roles),
        'recent_activity': list(recent_activity),
    }
    return JsonResponse(data, safe=False, encoder=DjangoJSONEncoder)
