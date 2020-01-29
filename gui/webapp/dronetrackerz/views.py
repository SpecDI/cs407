from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

#import static/todo/test
# Create your views here.

def droneView(request):
    #return HttpResponse('hello, this is todoView')
    context={}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        # print(uploaded_file.name)
        # print(uploaded_file.size)


        fs = FileSystemStorage()
        video_name = fs.save(uploaded_file.name, uploaded_file)
        print(video_name)
        context['url'] = fs.url(video_name)


        # call to pipeline with fs.url(name) as argument
        # return annoated video
        # set context to returned video
        # in this way will hang on website while processing and then will play


        #next step is to pretty it all up


    return render(request, 'drone.html', context)
