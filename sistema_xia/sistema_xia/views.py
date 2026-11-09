from __future__ import annotations

import json
from collections.abc import Iterator

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import Group, Permission
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from django.db.models import Q

from chats.models import Chats, Messages

from .agents.chatbot_agent import ChatbotAgent


PUBLIC_GROUP_NAME = "public_user"
PUBLIC_GROUP_PERMISSION_CODENAMES = [
    "add_chats",
    "change_chats",
    "delete_chats",
    "view_chats",
    "add_messages",
    "change_messages",
    "delete_messages",
    "view_messages",
]


def _attach_public_user_group(user) -> None:
    group, _ = Group.objects.get_or_create(name=PUBLIC_GROUP_NAME)
    permissions = list(Permission.objects.filter(codename__in=PUBLIC_GROUP_PERMISSION_CODENAMES))
    if permissions:
        group.permissions.add(*permissions)
    user.groups.add(group)


def register_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        terms_accepted = request.POST.get("terms_accepted")
        if not terms_accepted:
            messages.error(request, "Debe aceptar la Política de Privacidad para registrarse.")
        elif form.is_valid():
            user = form.save()
            _attach_public_user_group(user)
            login(request, user)
            messages.success(request, "Tu cuenta fue creada correctamente.")
            return redirect("home")
    else:
        form = UserCreationForm()

    return render(request, "auth/register.html", {"form": form})


@login_required
@require_GET
def chat_list(request):
    query = str(request.GET.get("q", "")).strip()
    chats = (
        Chats.objects.filter(user=request.user)
        .prefetch_related("messages")
        .order_by("-updated_at", "-created_at")
    )

    if query:
        chats = chats.filter(
            Q(title__icontains=query)
            | Q(messages__content__icontains=query)
        ).distinct().order_by("-updated_at", "-created_at")

    chats = chats[:10]

    payload = []
    for chat in chats:
        first_message = chat.messages.first()
        payload.append(
            {
                "id": chat.id,
                "title": chat.title or (first_message.content[:60] if first_message else f"Chat {chat.pk}"),
                "created_at": chat.created_at.isoformat(),
                "updated_at": chat.updated_at.isoformat(),
                "messages_count": chat.messages.count(),
                "display_date": chat.updated_at.strftime("%d/%m/%Y %H:%M"),
            }
        )

    return JsonResponse({"chats": payload})


@login_required
@require_GET
def chat_detail(request, chat_id: int):
    chat = get_object_or_404(Chats, pk=chat_id, user=request.user)
    messages_payload = [
        {
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat(),
        }
        for message in chat.messages.all().order_by("created_at")
    ]

    return JsonResponse(
        {
            "chat": {
                "id": chat.id,
                "title": chat.title,
                "created_at": chat.created_at.isoformat(),
                "updated_at": chat.updated_at.isoformat(),
            },
            "messages": messages_payload,
        }
    )


@login_required
@require_POST
def chat_delete(request, chat_id: int):
    chat = get_object_or_404(Chats, pk=chat_id, user=request.user)
    chat.delete()
    return JsonResponse({"deleted": True, "chat_id": chat_id})


def _chat_title_from_messages(conversation: list[dict[str, str]]) -> str:
    for item in conversation:
        if item.get("role") == "user" and item.get("content"):
            return item["content"][:60]
    return "Nuevo chat"


def _generate_chat_title(conversation: list[dict[str, str]], context: str | None = None) -> str:
    user_prompt = (
        "Genera un titulo corto para este chat. Reglas: devuelve solo el titulo, en español, "
        "maximo 6 palabras y 200 caracteres, sin comillas, sin puntuacion final, sin explicaciones. "
        "Debe ser especifico y resumir la primera consulta. Conversacion:\n"
    )
    transcript_lines = []
    for item in conversation:
        role = item.get("role", "")
        content = item.get("content", "")
        if role and content:
            transcript_lines.append(f"{role}: {content}")

    prompt = user_prompt + "\n".join(transcript_lines)
    try:
        raw_title = ChatbotAgent(temperature=0.0).generate(prompt, context=context, skip_audit=True)
    except Exception:
        return _chat_title_from_messages(conversation)

    title = " ".join(raw_title.replace("\n", " ").split()).strip("\"' .,:;!?-")
    if not title:
        return _chat_title_from_messages(conversation)

    words = title.split()
    if len(words) > 6:
        title = " ".join(words[:6])

    return title[:200]


def _replace_chat_messages(chat: Chats, conversation: list[dict[str, str]], assistant_reply: str) -> None:
    chat.messages.all().delete()
    messages_to_create = [
        Messages(chat=chat, role=item["role"], content=item["content"])
        for item in conversation
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    messages_to_create.append(Messages(chat=chat, role="assistant", content=assistant_reply))
    Messages.objects.bulk_create(messages_to_create)
    chat.title = chat.title or _chat_title_from_messages(conversation)
    chat.save(update_fields=["title", "updated_at"])


def _resolve_chat_for_request(
    request,
    conversation: list[dict[str, str]],
    chat_id: str | None,
    context: str | None = None,
) -> Chats | None:
    if not request.user.is_authenticated:
        return None

    if chat_id:
        return get_object_or_404(Chats, pk=chat_id, user=request.user)

    chat_title = _generate_chat_title(conversation, context=context)
    chat = Chats.objects.create(user=request.user, title=chat_title)
    return chat


@csrf_exempt
@require_POST
def chat_stream(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Carga JSON no válida."}, status=400)

    message = str(payload.get("message", "")).strip()
    raw_messages = payload.get("messages", [])
    raw_chat_id = payload.get("chat_id")
    chat_id = str(raw_chat_id).strip() if raw_chat_id not in {None, ""} else None
    context = payload.get("context")
    if context is None:
        context = ChatbotAgent().get_context()

    conversation: list[dict[str, str]] = []
    if isinstance(raw_messages, list):
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            conversation.append({"role": role, "content": content})

    if message and not conversation:
        conversation = [{"role": "user", "content": message}]

    if not conversation:
        return JsonResponse({"error": "Se requiere el historial de mensajes."}, status=400)

    conversation = conversation[-20:]
    chat = _resolve_chat_for_request(request, conversation, chat_id, context=context)

    def stream() -> Iterator[str]:
        agent = ChatbotAgent()
        try:
            complete_text = ""
            for chunk in agent.stream_chat_messages(conversation, context=context):
                complete_text += chunk
                yield json.dumps({"chunk": chunk}) + "\n"
            if chat is not None:
                with transaction.atomic():
                    _replace_chat_messages(chat, conversation, complete_text.strip() or "Sin respuesta del modelo.")
                yield json.dumps({"chat_id": chat.id}) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as exc:
            yield json.dumps({"error": str(exc), "done": True}) + "\n"

    response = StreamingHttpResponse(stream(), content_type="application/x-ndjson")
    response["Cache-Control"] = "no-cache"
    if chat is not None:
        response["X-Chat-Id"] = str(chat.id)
    return response
