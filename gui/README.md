# CS407 DroneTrackerz Async Web App

## Dependencies

 - Django (I used V2.1.15)
 - Celery (V4.4.2)
 - CeleryProgress (V0.0.9)
 - Redis (3.4.1)
 - Eventlet (Windows - V0.25.2)

## How to run

 - Install the above dependencies
 - Ensure that "_5_5_TransferLSTM_TS.hdf5" is in action_recognition/architectures/weights
 - Ensure that "yolo.h5" is in object_detection/model_data
 - Navigate to gui/webapp
 - Run: python manage.py runserver
 - In a separate Terminal window, navigate to the root level of the repo (i.e. same level as pipeline) and run: celery -A pipelineNewAsync worker --loglevel=info
 - If Windows: celery -A pipelineNewAsync worker --loglevel=info -P eventlet
 - In your browser, go to http://127.0.0.1:8000/drone/
 - Upload the desired (mp4) video and enjoy

 ## Note on Redis

I previously encountered an error where, after running celery -A webapp worker --loglevel=info, Celery was unable to connect to Redis.

In gui/webapp/webapp/settings.py, at the bottom are two URLs: CELERY_BROKER_URL and CELERY_RESULT_BACKEND.

The required URLs have changed for these once, and I'm not sure when/why this happens. If you get an error along the lines of: ERROR/MainProcess] consumer: Cannot connect to redis://h:**@ec2-52-208-3-240.eu-west-1.compute.amazonaws.com:7089//: Error 60 connecting to ec2-52-208-3-240.eu-west-1.compute.amazonaws.com:7089. Operation timed out..

Drop me a message and I'll find the update values for CELERY_BROKER_URL and CELERY_RESULT_BACKEND for you :)

Alternatively, Redis can be installed and run locally, but I haven't done this.
