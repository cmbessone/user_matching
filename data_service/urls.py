from django.urls import path
from . import views


from . import views
app_name = 'data_service'
urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
]
