"""
URL configuration for sistema_xia project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path, reverse_lazy
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static

from .views import chat_delete, chat_detail, chat_list, chat_stream, register_view

urlpatterns = [
    path('', TemplateView.as_view(template_name='chat.html'), name='home'),
    path('accounts/login/', LoginView.as_view(template_name='registration/login.html', redirect_authenticated_user=True), name='login'),
    path('accounts/logout/', LogoutView.as_view(next_page=reverse_lazy('home')), name='logout'),
    path('accounts/register/', register_view, name='register'),
    path('api/chats/', chat_list, name='chat_list'),
    path('api/chats/<int:chat_id>/', chat_detail, name='chat_detail'),
    path('api/chats/<int:chat_id>/delete/', chat_delete, name='chat_delete'),
    path('api/chat/stream/', chat_stream, name='chat_stream'),
    path('admin/', admin.site.urls),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

