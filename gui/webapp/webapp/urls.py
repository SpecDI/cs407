#from django.conf.urls import patterns, include, url
#from django.conf.urls.defaults import *
from django.conf.urls import include, url

from django.contrib import admin

from dronetrackerz.views import droneView
#from dronetrackerz.views import uploadView
from dronetrackerz.views import progressView


from django.conf import settings
from django.conf.urls.static import static

from django.urls import re_path

admin.autodiscover()



# urlpatterns = patterns('',
#     # Examples:
#     # url(r'^$', 'webapp.views.home', name='home'),
#     # url(r'^blog/', include('blog.urls')),
#
#     url(r'^admin/', include(admin.site.urls)),
#     url(r'^drone/', droneView),
# )

urlpatterns = [
    url(r'^drone/', droneView),
    #url('upload/', uploadView, name='upload'),
    #url('http://127.0.0.1:8000/drone/drone_progress.html', progressView),
    url('', include('dronetrackerz.urls')),
    re_path(r'^celery-progress/', include('celery_progress.urls')),
    # ... your url patterns
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
