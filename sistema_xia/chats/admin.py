from django.contrib import admin
from django.urls import path
from .models import Chats, Messages
from .views import chat_dashboard_api

class MessagesInline(admin.TabularInline):
	model = Messages
	extra = 0
	fields = ("role", "content", "created_at")
	readonly_fields = ("created_at",)


@admin.register(Chats)
class ChatsAdmin(admin.ModelAdmin):
	list_display = ("id", "user", "title", "created_at", "updated_at")
	list_filter = ("created_at", "updated_at")
	search_fields = ("title", "user__username", "user__email")
	inlines = [MessagesInline]
	change_list_template = "admin/chats/chats_changelist.html"

	def get_urls(self):
		urls = super().get_urls()
		custom_urls = [
			path('api/dashboard-data/', self.admin_site.admin_view(chat_dashboard_api), name='chat-dashboard-data'),
		]
		return custom_urls + urls

@admin.register(Messages)
class MessagesAdmin(admin.ModelAdmin):
	list_display = ("id", "chat", "role", "created_at")
	list_filter = ("role", "created_at")
	search_fields = ("content", "chat__title", "chat__user__username")

# Register your models here.
