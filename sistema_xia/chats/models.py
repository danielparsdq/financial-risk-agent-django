from django.conf import settings
from django.db import models


class Chats(models.Model):
	user = models.ForeignKey(
		settings.AUTH_USER_MODEL,
		on_delete=models.CASCADE,
		related_name="chats",
	)
	title = models.CharField(max_length=200, blank=True)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ["-updated_at", "-created_at"]
		verbose_name = "Chat"
		verbose_name_plural = "Chats"

	def __str__(self) -> str:
		return self.title or f"Chat {self.pk}"


class Messages(models.Model):
	class Role(models.TextChoices):
		USER = "user", "Usuario"
		ASSISTANT = "assistant", "Asistente"
		SYSTEM = "system", "Sistema"

	chat = models.ForeignKey(Chats, on_delete=models.CASCADE, related_name="messages")
	role = models.CharField(max_length=20, choices=Role.choices)
	content = models.TextField()
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ["created_at"]
		verbose_name = "Mensaje"
		verbose_name_plural = "Mensajes"

	def __str__(self) -> str:
		return f"{self.role} @ {self.created_at:%Y-%m-%d %H:%M:%S}" if self.created_at else self.role

# Create your models here.
