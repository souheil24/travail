from django.contrib import admin
from django.urls import path, include
from accounts.views import landing_page
from accounts.views import landing_page, analyze_all_comments , add_comment_Demo , analyze_all_comments_FB 
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', landing_page, name='home'),
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('add_comment_Demo/', add_comment_Demo , name='add_comment'),
    path('predict/', analyze_all_comments , name='predict_view'),
    path('predictFB/', analyze_all_comments_FB , name='predict_view'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)