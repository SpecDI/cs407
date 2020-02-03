from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

# Create your views here.



def droneView(request):
    #return HttpResponse('hello, this is todoView')

    from static.test import run_test
    #run_test()

    context={}
    if request.method == 'POST':
        from static.test import modify_video
        uploaded_file = request.FILES['document']
        # print(uploaded_file.name)
        # print(uploaded_file.size)


        fs = FileSystemStorage()
        video_name = fs.save(uploaded_file.name, uploaded_file)
        print("Video name: ", video_name)
        context['url'] = fs.url(video_name) #NEED TO REMOVE LATER
        print("Video URL: ", fs.url(video_name))

        #modify_video(fs.url(video_name))

        #modify_video('/media/1.1.1.mov')
        #modify_video('1.1.1.mov')
        #context['url'] = 'edited_video.mp4'



        # call to pipeline with fs.url(name) as argument
        # return annoated video
        # set context to returned video
        # in this way will hang on website while processing and then will play


        #next step is to pretty it all up


    return render(request, 'drone.html', context)
