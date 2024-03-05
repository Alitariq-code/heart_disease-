from django.urls import path
from .views import Pre1  # Ensure to import your view
from .views import Pre2  # Ensure to import your view

urlpatterns = [
    path('Pre1/', Pre1.as_view(), name='Pre1'),
    path('Pre2/', Pre2.as_view(), name='Pre2'),
]