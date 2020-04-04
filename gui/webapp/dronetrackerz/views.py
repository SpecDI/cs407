from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

from .tasks import sleepy, dummy_modify_video
from time import sleep

import os.path
from os import path

# Create your views here.

# def index(request):
#     sleepy.delay(10)
#     return HttpResponse('<h1>TASK IS DONE!</h1>')



# Just render the basic upload landing page.
# def uploadView(request):
#     if request.method == 'POST':
#         uploaded_file = request.FILES['document']
#         print(uploaded_file.name)
#         print(uploaded_file.size)
#     return render(request, 'drone_upload.html')


# Kick off Celery, display progress bar
def progressView(request):
    context = {}

    uploaded_file = request.FILES['document']

    fs = FileSystemStorage()
    video_name = fs.save(uploaded_file.name, uploaded_file)
    # context['url'] = fs.url(video_name) #NEED TO REMOVE LATER
    print("Video name: ", video_name)
    #print("Video URL: ", fs.url(video_name))
    video_url = fs.url(video_name)
    print("Video URL: ", video_url)

    result = dummy_modify_video.delay(video_url)

    return render(request, 'drone_progress.html', context={'task_id': result.task_id})


# Main
def droneView(request):
    context={}
    if request.method == 'POST':
        from static.test import modify_video
        uploaded_file = request.FILES['document']

        fs = FileSystemStorage()
        # Delete previously uploaded input file (if it exists)
        fs.delete('1.1.1.mov')
        video_name = fs.save(uploaded_file.name, uploaded_file)
        print("Video name: " + video_name)
        #context['url'] = fs.url(video_name) #NEED TO REMOVE LATER
        video_url = fs.url(video_name)
        print("Video URL: " + video_url)

        video_url = video_url[1:]
        print("Chopped VideoURL: ", video_url)

        print("Video_url data type: " , type(video_url))
        result = dummy_modify_video.delay(video_url)


        return render(request, 'drone_progress.html', context={'task_id': result.task_id})

    
    return render(request, 'drone_upload.html', context)


