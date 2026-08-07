"""
URL configuration for hermes_ecossistema project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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
from django.urls import path
from hermes_cad import views 


urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('home/', views.home, name='home'),
    path('emergency_info/', views.emergency_info, name='emergency_info'),
    path('emergencias/', views.emergencias, name='emergencias'),


    path('proxy/list_emergencies/', views.proxy_emergencias, name='proxy_emergencias'),
    path('proxy/start_call_interpretation/', views.proxy_start_call_interpretation, name='proxy_start_call_interpretation'),
    path('proxy/end_call/', views.proxy_end_call, name='proxy_end_call'),
    path('proxy/send_transcript/', views.proxy_send_transcript, name='proxy_send_transcript'),
    path('proxy/get_all_inference_results/', views.proxy_get_all_inference_results, name='proxy_get_all_inference_results'),
    # TODO: re_path(r'^proxy/(?P<path>.*)$', proxy)

    # URLs da API
    # path('api/', views.UsuarioViewSet.as_view({'get': 'list', 'post': 'create'}), name='api_usuarios'),
    # path('api/<int:pk>/', views.UsuarioViewSet.as_view({'get': 'retrieve', 'put': 'update', 'patch': 'partial_update', 'delete': 'destroy'}), name='api_usuarios_detail'),
    
]
