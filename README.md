# Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA
Hands-On GPU Accelerated Computer Vision with OpenCV and CUDA, published by Packt

<a href="https://www.packtpub.com/application-development/hands-gpu-accelerated-computer-vision-opencv-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781789348293 "><img src="https://d255esdrn735hr.cloudfront.net/sites/default/files/imagecache/ppv4_main_book_cover/cover%20-%20Copy_10995.png" alt="Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA" height="256px" align="right"></a>

This is the code repository for [Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA](https://www.packtpub.com/application-development/hands-gpu-accelerated-computer-vision-opencv-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781789348293 ), published by Packt.

**Effective techniques for processing complex image data in real time using GPUs	**

## What is this book about?
Computer vision has been revolutionizing a wide range of industries, and OpenCV is the most widely chosen tool for computer vision with its ability to work in multiple programming languages. Nowadays, in computer vision, there is a need to process large images in real time, which is difficult to handle for OpenCV on its own. This is where CUDA comes into the picture, allowing OpenCV to leverage powerful NVDIA GPUs. This book provides a detailed overview of integrating OpenCV with CUDA for practical applications.

This book covers the following exciting features:
Understand how to access GPU device properties and capabilities from CUDA programs
<ul>
    <li>Learn how to accelerate searching and sorting algorithms</li>
    
<li>Detect shapes such as lines and circles in images</li>

<li>Explore object tracking and detection with algorithms</li>

<li>Process videos using different video analysis techniques in Jetson TX1</li>

<li>Access GPU device properties from the PyCUDA program</li>

<li>Understand how kernel execution works </li>

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1789348293) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
while (tid < N)
    {
       d_c[tid] = d_a[tid] + d_b[tid];
       tid += blockDim.x * gridDim.x;
    }
```

**Following is what you need for this book:**
This book is a go-to guide for you if you are a developer working with OpenCV and want to learn how to process more complex image data by exploiting GPU processing. A thorough understanding of computer vision concepts and programming languages such as C++ or Python is expected.

With the following software and hardware list you can run all code files present in the book (Chapter 1-12).
### Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| 1-4 | CUDA Toolkit X.X, Microsoft Visual Studio Community Edition, Nsight | Windows, Mac OS X, and Linux (Any) |
| 5-8 | OpenCV Library | Windows, Mac OS X, and Linux (Any) |
| 10-12 | Anaconda Python, PyCUDA | Windows, Mac OS X, and Linux (Any) |


We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://www.packtpub.com/sites/default/files/downloads/978-1-78934-829-3_ColorImages.pdf).

Visit the following link to check out videos of the code being run: http://bit.ly/2PZOYcH

### Related products
* OpenCV 3 Computer Vision with Python Cookbook [[Packt]](https://www.packtpub.com/application-development/opencv-3-computer-vision-python-cookbook?utm_source=github&utm_medium=repository&utm_campaign=) [[Amazon]](https://www.amazon.com/dp/1788474449)

* Computer Vision with OpenCV 3 and Qt5 [[Packt]](https://www.packtpub.com/application-development/computer-vision-opencv-3-and-qt5?utm_source=github&utm_medium=repository&utm_campaign=9781788472395 ) [[Amazon]](https://www.amazon.com/dp/178847239X)


## Get to Know the Author
**Bhaumik Vaidya**
Bhaumik Vaidya is an experienced computer vision engineer and mentor. He has worked extensively on OpenCV Library in solving computer vision problems. He is a University gold medalist in masters and is now doing a PhD in the acceleration of computer vision algorithms built using OpenCV and deep learning libraries on GPUs. He has a background in teaching and has guided many projects in computer vision and VLSI(Very-large-scale integration). He has worked in the VLSI domain previously as an ASIC verification engineer, so he has very good knowledge of hardware architectures also. He has published many research papers in reputable journals to his credit. He, along with his PhD mentor, has also received an NVIDIA Jetson TX1 embedded development platform as a research grant from NVIDIA.



### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.


