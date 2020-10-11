# Motion Detection and Analysis from Drone Footage using Deep Learning Techniques

***Awarded Best MEng dissertation by the Department of Computer Science, University of Warwick.***

The project sets out to achieve real-time motion detection and action recognition from drone footage using a variety of deep learning techniques.

![](demo_img.png)

The system can identify 12 unique actions; from running and walking to handshaking and hugging. The system performs well in the real-world challenge of multi-labelled actors where an actor performs more than one action at the same time and in the dynamic transition of actions where actors sequentially perform a diverse set of actions. The underlying system can be viewed as a pipeline split into three components; object detection, object tracking, and action recognition. Our object detection and tracking components build on existing architectures; with innovative optimisations designed specifically for the challenge of continuous tracking of humans from a moving drone in real-time. Our action-recognition component provides an entirely new architecture, reinventing how high performance can be achieved through transfer learning, temporal pooling and a variational LSTM to perform Bayesian inference. Through rigorous testing, both on individual components and the entire system, we have been able to produce an architecture which not only performs real-time motion detection and action recognition in real-time but also outperforms world-leading papers in the field on a variety of metrics (including hamming loss, frames-per-second etc.).
