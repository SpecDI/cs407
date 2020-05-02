from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

from .tasks import sleepy, dummy_modify_video
from time import sleep

import os.path
from os import path

from celery.execute import send_task

import re


# Kick off Celery, display progress bar
#NOT USED
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

        print("=============================================")
        fs = FileSystemStorage()
        items = fs.listdir("")
        print("Items: ", items)
        poss_mp4s = items[1]
        r = re.compile(".*mp4")
        true_mp4s = list(filter(r.match, poss_mp4s))
        print("True MP4s: " , true_mp4s)
        for vid in true_mp4s:
            print("Deleting:" , vid)
            fs.delete(vid)
        print("=============================================")




        # Delete previously uploaded input file (if it exists)
        video_name = fs.save(uploaded_file.name, uploaded_file)
        print("Video name: " + video_name)
        #context['url'] = fs.url(video_name) #NEED TO REMOVE LATER
        video_url = fs.url(video_name)
        print("Video URL: " + video_url)

        video_url = video_url[1:]
        print("Chopped VideoURL: ", video_url)

        print("Video_url data type: " , type(video_url))
        #result = dummy_modify_video.delay(video_url)
        #result = send_task("tasks.dummy_modify_video" , kwargs=dict(value="video_url") ) 
        #result = send_task("tasks.dummy_modify_video" , args=[video_url] ) 
        result = send_task("pipeline_async", args=[video_url])

        return render(request, 'drone_progress.html', context={'task_id': result.task_id})

    
    return render(request, 'drone_upload.html', context)


